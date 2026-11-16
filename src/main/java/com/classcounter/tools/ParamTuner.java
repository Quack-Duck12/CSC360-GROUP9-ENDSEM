package com.classcounter.tools;

import com.classcounter.util.NativeLoader;
import com.classcounter.vision.HeadDetector;
import com.classcounter.vision.HeadDetector.*;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.w3c.dom.*;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

/**
 * Offline tool (not part of the app): searches {@link HeadDetector.Params} against SCUT-HEAD ground truth
 * (Pascal VOC XML) with random sampling followed by hill climbing, and prints the best parameters as Java code.
 * Scores are computed on tuning images and on held-out images the search never saw.
 *
 * <p>Usage (arguments come in {@code --name value} pairs):
 * <pre>{@code
 * --root <dir>          folder containing JPEGImages and Annotations (SCUT-HEAD Part B)
 * --split train|val|test  image list under ImageSets (default train)
 * --n 60 --holdout 30   number of tuning and held-out images
 * --iters 300           search iterations
 * --minutes 30          optional time limit
 * --objective f1|count|mix   what to maximise (mix = F1 * (1 - count error))
 * --match-iou 0.5       IoU needed to count a detection as correct
 * --method threshold|edges|both|hough|blob   search only this method
 * --min-gt-area 100     ignore ground-truth heads smaller than this (px^2)
 * --seed 1
 * }</pre>
 */
public class ParamTuner {

    /**
     * One evaluation image: the downscaled picture and its ground-truth head boxes.
     */
    record Sample(String name, Mat image, List<Rect> gt) {}

    static final int MAX_WIDTH = 1280;
    static double minGtArea = 100;
    static Method forced = null;     // --method: search only this method
    static double matchIou = 0.5;   // IoU needed to count a detection as a hit
    static String mode = "f1";   // f1 | count | mix

    /**
     * Entry point: loads the data, runs the search and prints the summary and the Java block to paste.
     */
    public static void main(String[] a) throws Exception {
        Map<String, String> o = parse(a);
        if (!o.containsKey("root")) { System.err.println("Missing --root <dir containing JPEGImages and Annotations>"); return; }
        NativeLoader.load();
        int n = Integer.parseInt(o.getOrDefault("n", "60"));
        int hold = Integer.parseInt(o.getOrDefault("holdout", "30"));
        int iters = Integer.parseInt(o.getOrDefault("iters", "300"));
        long seed = Long.parseLong(o.getOrDefault("seed", "1"));
        minGtArea = Double.parseDouble(o.getOrDefault("min-gt-area", "100"));
        mode = o.getOrDefault("objective", "f1");
        matchIou = Double.parseDouble(o.getOrDefault("match-iou", "0.5"));
        if (o.containsKey("method")) forced = Method.valueOf(o.get("method").toUpperCase());

        List<String> names = listNames(Path.of(o.get("root")), o.getOrDefault("split", "train"));
        Collections.shuffle(names, new Random(seed));
        if (names.size() < n + hold) { n = Math.max(1, names.size() * 2 / 3); hold = names.size() - n; }
        System.out.printf("Found %d images; tuning on %d, held-out %d%n", names.size(), n, hold);
        List<Sample> tune = load(Path.of(o.get("root")), names.subList(0, n));
        List<Sample> held = load(Path.of(o.get("root")), names.subList(n, n + hold));

        Random rnd = new Random(seed);
        Params best = new Params();
        if (forced != null) best.method = forced;
        System.out.println("Evaluating defaults ...");
        Score bestScore = eval(tune, best);
        System.out.println("Defaults        : " + bestScore);
        List<Sample> quick = tune.subList(0, Math.min(15, tune.size()));
        Score quickBest = eval(quick, best);
        double margin = mode.equals("count") ? 0.05 : mode.equals("mix") ? 0.01 : 0.02;
        long start = System.currentTimeMillis();
        long deadline = o.containsKey("minutes")
                ? start + (long) (Double.parseDouble(o.get("minutes")) * 60_000) : Long.MAX_VALUE;

        for (int i = 1; i <= iters; i++) {
            // first 40% random exploration, rest hill climbing around the best
            Params cand = i <= iters * 0.4 ? random(rnd) : mutate(best, rnd);
            if (System.currentTimeMillis() > deadline) {
                System.out.println("Time limit reached after " + (i - 1) + " iterations");
                break;
            }
            if (i % 10 == 0) System.out.printf("  ... %d/%d iterations done (%.0f s elapsed)%n",
                    i, iters, (System.currentTimeMillis() - start) / 1000.0);
            Score q = eval(quick, cand);                       // cheap check on a few images first
            if (q.objective() < quickBest.objective() - margin) continue;   // clearly worse: skip it
            Score s = eval(tune, cand);
            System.gc();   // lets OpenCV free temporary native Mats promptly
            if (s.objective() > bestScore.objective()) {
                best = cand; bestScore = s; quickBest = q;
                System.out.printf("iter %3d  better : %s%n", i, s);
            }
        }
        System.out.println("\nBest on tuning set : " + bestScore);
        if (!held.isEmpty()) {
            System.out.println("Defaults on held-out: " + eval(held, new Params()));
            System.out.println("Best on held-out    : " + eval(held, best));
        }
        saveDebug(held.isEmpty() ? tune : held, best, Path.of(o.getOrDefault("debug-dir", "tuner-debug")));
        System.out.println("\n--- paste into HeadDetector.Params ---");
        System.out.println(asJava(best));
    }

    // ---------- evaluation ----------

    /**
     * Detection quality over a set of images.
     *
     * @param tp       detections that matched a ground-truth head
     * @param fp       detections that matched nothing
     * @param fn       ground-truth heads that were missed
     * @param countErr mean over images of |detected - true| / true
     */
    record Score(int tp, int fp, int fn, double countErr) {
        double precision() { return tp + fp == 0 ? 0 : tp / (double) (tp + fp); }
        double recall() { return tp + fn == 0 ? 0 : tp / (double) (tp + fn); }
        double f1() { double p = precision(), r = recall(); return p + r == 0 ? 0 : 2 * p * r / (p + r); }
        double objective() { return switch (mode) { case "count" -> -countErr; case "mix" -> f1() * Math.max(0, 1 - countErr); default -> f1(); }; }
        public String toString() {
            return String.format("F1=%.3f  P=%.3f  R=%.3f  mean|count error|=%.1f%%", f1(), precision(), recall(), countErr * 100);
        }
    }

    /**
     * Scores one parameter set on a set of images (images are processed in parallel).
     * A detection is correct when its IoU with a not-yet-used ground-truth box reaches {@code matchIou}.
     */
    static Score eval(List<Sample> set, Params p) {
        HeadDetector det = new HeadDetector();
        int[] tot = new int[3];
        double[] err = new double[1];
        {
            set.parallelStream().forEach(s -> {
                Result res = det.detect(s.image(), p);
                List<Box> pred = res.boxes();
                res.gray().release(); res.blurred().release(); res.edges().release();
                res.binary().release(); res.cleaned().release();   // free native memory right away
                boolean[] used = new boolean[s.gt().size()];
                int tp = 0;
                for (Box b : pred) {
                    int bi = -1; double bv = matchIou;
                    for (int g = 0; g < used.length; g++) {
                        if (used[g]) continue;
                        double v = iou(b.rect(), s.gt().get(g));
                        if (v >= bv) { bv = v; bi = g; }
                    }
                    if (bi >= 0) { used[bi] = true; tp++; }
                }
                int fp = pred.size() - tp, fn = s.gt().size() - tp;
                double ce = s.gt().isEmpty() ? 0 : Math.abs(pred.size() - s.gt().size()) / (double) s.gt().size();
                synchronized (tot) { tot[0] += tp; tot[1] += fp; tot[2] += fn; err[0] += ce; }
            });
        }
        return new Score(tot[0], tot[1], tot[2], err[0] / Math.max(1, set.size()));
    }

    /**
     * Intersection over union of two rectangles, 0 when disjoint.
     */
    static double iou(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width), y2 = Math.min(a.y + a.height, b.y + b.height);
        double inter = Math.max(0, x2 - x1) * (double) Math.max(0, y2 - y1);
        double uni = a.area() + b.area() - inter;
        return uni <= 0 ? 0 : inter / uni;
    }

    // ---------- search space ----------

    /**
     * Draws a random parameter set; the method is fixed when {@code --method} was given.
     */
    static Params random(Random r) {
        Params p = new Params();
        p.method = forced != null ? forced : Method.values()[r.nextInt(Method.values().length)];
        p.threshold = Threshold.values()[r.nextInt(3)];
        p.blur = 1 + 2 * r.nextInt(6);
        p.globalValue = 40 + r.nextInt(140);
        p.adaptiveBlock = 11 + 2 * r.nextInt(46);
        p.adaptiveC = 2 + r.nextInt(24);
        p.cannyLow = 20 + r.nextInt(100);
        p.cannyHigh = p.cannyLow + 20 + r.nextInt(150);
        p.morph = 1 + r.nextInt(7);
        p.minArea = 20 + r.nextInt(780);
        p.maxArea = p.minArea + 300 + r.nextInt(7000);
        p.minAspect = 0.3 + r.nextDouble() * 0.6;
        p.maxAspect = 1.2 + r.nextDouble() * 1.0;
        p.minCircularity = r.nextDouble() * 0.7;
        p.iou = 0.1 + r.nextDouble() * 0.5;
        p.houghParam2 = 10 + r.nextInt(50);
        p.houghMinR = 3 + r.nextInt(15);
        p.houghMaxR = p.houghMinR + 5 + r.nextInt(30);
        p.houghMinDist = 5 + r.nextInt(30);
        p.blobThreshold = 15 + r.nextInt(70);
        p.blobRTop = 3 + r.nextInt(12);
        p.blobRBottom = p.blobRTop + r.nextInt(30);
        p.blobTol = 0.3 + r.nextDouble() * 0.9;
        p.minSolidity = r.nextDouble() < 0.5 ? 0 : 0.3 + r.nextDouble() * 0.6;
        p.maxEccentricity = r.nextDouble() < 0.5 ? 10 : 1.5 + r.nextDouble() * 4;
        return p;
    }

    /**
     * Copies every field of a parameter set.
     */
    static Params copy(Params s) {
        Params p = new Params();
        p.method = s.method; p.threshold = s.threshold; p.blur = s.blur; p.globalValue = s.globalValue;
        p.adaptiveBlock = s.adaptiveBlock; p.adaptiveC = s.adaptiveC; p.cannyLow = s.cannyLow;
        p.cannyHigh = s.cannyHigh; p.morph = s.morph; p.minArea = s.minArea; p.maxArea = s.maxArea;
        p.minAspect = s.minAspect; p.maxAspect = s.maxAspect; p.minCircularity = s.minCircularity; p.iou = s.iou;
        p.houghParam2 = s.houghParam2; p.houghMinR = s.houghMinR; p.houghMaxR = s.houghMaxR; p.houghMinDist = s.houghMinDist;
        p.minSolidity = s.minSolidity; p.maxEccentricity = s.maxEccentricity;
        p.blobThreshold = s.blobThreshold; p.blobRTop = s.blobRTop; p.blobRBottom = s.blobRBottom; p.blobTol = s.blobTol;
        return p;
    }

    /** Changes 1-3 random parameters by a small step. */
    static Params mutate(Params s, Random r) {
        Params p = copy(s);
        int changes = 1 + r.nextInt(3);
        for (int c = 0; c < changes; c++) {
            switch (r.nextInt(23)) {
                case 0 -> p.blur = clamp(p.blur + 2 * (r.nextInt(3) - 1), 1, 15);
                case 1 -> p.globalValue = clamp(p.globalValue + r.nextInt(31) - 15, 0, 255);
                case 2 -> p.adaptiveBlock = clamp(p.adaptiveBlock + 2 * (r.nextInt(7) - 3), 3, 101);
                case 3 -> p.adaptiveC = clamp(p.adaptiveC + r.nextInt(5) - 2, 0, 30);
                case 4 -> p.cannyLow = clamp(p.cannyLow + r.nextInt(21) - 10, 5, 200);
                case 5 -> p.cannyHigh = clamp(p.cannyHigh + r.nextInt(31) - 15, p.cannyLow, 255);
                case 6 -> p.morph = clamp(p.morph + r.nextInt(3) - 1, 1, 9);
                case 7 -> p.minArea = Math.max(10, p.minArea * (0.8 + r.nextDouble() * 0.4));
                case 8 -> p.maxArea = Math.max(p.minArea + 100, p.maxArea * (0.8 + r.nextDouble() * 0.4));
                case 9 -> p.minAspect = Math.min(p.maxAspect - 0.2, Math.max(0.1, p.minAspect + r.nextDouble() * 0.2 - 0.1));
                case 10 -> p.maxAspect = Math.max(p.minAspect + 0.2, p.maxAspect + r.nextDouble() * 0.3 - 0.15);
                case 11 -> p.minCircularity = Math.min(0.9, Math.max(0, p.minCircularity + r.nextDouble() * 0.2 - 0.1));
                case 12 -> p.houghParam2 = clamp(p.houghParam2 + r.nextInt(7) - 3, 8, 80);
                case 13 -> p.houghMinR = clamp(p.houghMinR + r.nextInt(5) - 2, 2, Math.max(2, p.houghMaxR - 2));
                case 14 -> p.houghMaxR = clamp(p.houghMaxR + r.nextInt(9) - 4, p.houghMinR + 2, 80);
                case 15 -> p.houghMinDist = clamp(p.houghMinDist + r.nextInt(7) - 3, 3, 60);
                case 16 -> p.minSolidity = Math.min(0.95, Math.max(0, p.minSolidity + r.nextDouble() * 0.2 - 0.1));
                case 17 -> p.maxEccentricity = Math.min(10, Math.max(1.3, p.maxEccentricity + r.nextDouble() * 1.0 - 0.5));
                case 18 -> p.blobThreshold = clamp(p.blobThreshold + r.nextInt(9) - 4, 10, 150);
                case 19 -> p.blobRTop = clamp(p.blobRTop + r.nextInt(5) - 2, 2, 40);
                case 20 -> p.blobRBottom = clamp(p.blobRBottom + r.nextInt(7) - 3, 2, 80);
                case 21 -> p.blobTol = Math.min(1.5, Math.max(0.1, p.blobTol + r.nextDouble() * 0.2 - 0.1));
                default -> p.iou = Math.min(0.8, Math.max(0.05, p.iou + r.nextDouble() * 0.2 - 0.1));
            }
        }
        if (p.cannyHigh < p.cannyLow) p.cannyHigh = p.cannyLow;
        return p;
    }

    /** Saves overlays: ground truth in red, detections in green, so mismatches are visible. */
    static void saveDebug(List<Sample> set, Params p, Path dir) throws Exception {
        Files.createDirectories(dir);
        HeadDetector det = new HeadDetector();
        for (int i = 0; i < Math.min(8, set.size()); i++) {
            Sample sm = set.get(i);
            List<Box> pred = det.detect(sm.image(), p).boxes();
            Mat out = sm.image().clone();
            for (Rect g : sm.gt()) Imgproc.rectangle(out, g.tl(), g.br(), new Scalar(0, 0, 255), 1);
            for (Box b : pred) Imgproc.rectangle(out, b.rect().tl(), b.rect().br(), new Scalar(0, 255, 0), 2);
            Imgcodecs.imwrite(dir.resolve(sm.name() + ".png").toString(), out);
            System.out.printf("debug %s: ground truth %d, detected %d%n", sm.name(), sm.gt().size(), pred.size());
        }
        System.out.println("Debug images saved to " + dir.toAbsolutePath());
    }

    /**
     * Limits a value to the range lo..hi.
     */
    static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }

    /**
     * Formats all parameters as Java field lines ready to paste into {@code HeadDetector.Params}.
     */
    static String asJava(Params p) {
        return asJavaBase(p) + String.format(Locale.ROOT, "%npublic int houghParam2 = %d;%npublic int houghMinR = %d, houghMaxR = %d;%npublic int houghMinDist = %d;%npublic double minSolidity = %.2f, maxEccentricity = %.2f;%npublic int blobThreshold = %d, blobRTop = %d, blobRBottom = %d;%npublic double blobTol = %.2f;",
                p.houghParam2, p.houghMinR, p.houghMaxR, p.houghMinDist, p.minSolidity, p.maxEccentricity, p.blobThreshold, p.blobRTop, p.blobRBottom, p.blobTol);
    }

    /**
     * Formats the first group of parameters (method, thresholds, areas, shape limits).
     */
    static String asJavaBase(Params p) {
        return String.format(Locale.ROOT, String.join("\n",
                "public Method method = Method.%s;", "public Threshold threshold = Threshold.%s;",
                "public int blur = %d;", "public int globalValue = %d;", "public int adaptiveBlock = %d;",
                "public int adaptiveC = %d;", "public int cannyLow = %d, cannyHigh = %d;", "public int morph = %d;",
                "public double minArea = %.0f, maxArea = %.0f;", "public double minAspect = %.2f, maxAspect = %.2f;",
                "public double minCircularity = %.2f;", "public double iou = %.2f;"),
                p.method, p.threshold, p.blur, p.globalValue, p.adaptiveBlock, p.adaptiveC, p.cannyLow, p.cannyHigh,
                p.morph, p.minArea, p.maxArea, p.minAspect, p.maxAspect, p.minCircularity, p.iou);
    }

    // ---------- dataset (SCUT-HEAD, Pascal VOC layout) ----------

    /**
     * Lists the image names of a split. Uses {@code ImageSets/Main/<split>.txt} (or {@code ImageSets/<split>.txt}) when
     * present, otherwise every annotated image.
     */
    static List<String> listNames(Path root, String split) throws Exception {
        Path ann = find(root, "Annotations"), img = find(root, "JPEGImages");
        Path list = null;
        for (String rel : new String[]{"ImageSets/Main/" + split + ".txt", "ImageSets/" + split + ".txt"}) {
            Path c = ann.resolveSibling(rel);
            if (Files.exists(c)) { list = c; break; }
        }
        List<String> names;
        if (list != null && Files.exists(list)) {
            names = Files.readAllLines(list).stream().map(String::trim).filter(s -> !s.isEmpty())
                    .map(s -> s.split("\\s+")[0]).collect(Collectors.toList());
            System.out.println("Using split list " + list);
        } else {
            try (Stream<Path> s = Files.list(ann)) {
                names = s.map(x -> x.getFileName().toString()).filter(x -> x.endsWith(".xml"))
                        .map(x -> x.substring(0, x.length() - 4)).collect(Collectors.toList());
            }
            System.out.println("No ImageSets list found; using all annotated images");
        }
        Path imgDir = img;
        return names.stream().filter(nm -> Files.exists(ann.resolve(nm + ".xml")) && imageFile(imgDir, nm) != null)
                .collect(Collectors.toList());
    }

    /**
     * Finds a sub-folder with the given name up to four levels below the root.
     */
    static Path find(Path root, String dirName) throws Exception {
        try (Stream<Path> s = Files.walk(root, 4)) {
            return s.filter(Files::isDirectory).filter(d -> d.getFileName().toString().equals(dirName)).findFirst().orElse(null);
        }
    }

    /**
     * Finds the image file for a name, trying common extensions; null when missing.
     */
    static Path imageFile(Path imgDir, String name) {
        for (String ext : new String[]{".jpg", ".jpeg", ".JPG", ".png"}) {
            Path p = imgDir.resolve(name + ext);
            if (Files.exists(p)) return p;
        }
        return null;
    }

    /**
     * Loads images and VOC boxes. Images wider than {@code MAX_WIDTH} are shrunk like the app does and the boxes are
     * scaled the same way; ground-truth heads below {@code minGtArea} are ignored.
     */
    static List<Sample> load(Path root, List<String> names) throws Exception {
        Path ann = find(root, "Annotations"), img = find(root, "JPEGImages");
        List<Sample> out = new ArrayList<>();
        for (String nm : names) {
            Mat m = Imgcodecs.imread(imageFile(img, nm).toString(), Imgcodecs.IMREAD_COLOR);
            if (m.empty()) continue;
            double s = m.cols() > MAX_WIDTH ? MAX_WIDTH / (double) m.cols() : 1.0;
            if (s != 1.0) Imgproc.resize(m, m, new Size(), s, s, Imgproc.INTER_AREA);
            Document d = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(ann.resolve(nm + ".xml").toFile());
            NodeList objs = d.getElementsByTagName("bndbox");
            List<Rect> gt = new ArrayList<>();
            for (int i = 0; i < objs.getLength(); i++) {
                Element e = (Element) objs.item(i);
                int x1 = (int) Math.round(num(e, "xmin") * s), y1 = (int) Math.round(num(e, "ymin") * s);
                int x2 = (int) Math.round(num(e, "xmax") * s), y2 = (int) Math.round(num(e, "ymax") * s);
                Rect r = new Rect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
                if (r.area() >= minGtArea) gt.add(r);   // ignore heads too tiny to be detectable
            }
            out.add(new Sample(nm, m, gt));
        }
        return out;
    }

    /**
     * Reads a numeric child element such as xmin.
     */
    static double num(Element e, String tag) {
        return Double.parseDouble(e.getElementsByTagName(tag).item(0).getTextContent().trim());
    }

    /**
     * Parses {@code --name value} argument pairs into a map.
     */
    static Map<String, String> parse(String[] a) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i + 1 < a.length; i += 2) m.put(a[i].replaceFirst("^--", ""), a[i + 1]);
        return m;
    }
}
