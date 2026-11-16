package com.classcounter.vision;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.*;

/**
 * Classical head-candidate detector: only image-processing
 * operations implemented by OpenCV. It does <em>not</em> recognise heads. It finds dark, roughly round
 * regions (or blobs) of plausible size and shape and counts each one as a "head candidate".
 *
 * <p>Textbook mapping (Burger &amp; Burge, <em>Digital Image Processing</em>, 2016):
 * <ul>
 *   <li>grayscale conversion and Gaussian smoothing: Ch. 4 and 5.2</li>
 *   <li>Canny edges: Ch. 6.5</li>
 *   <li>global, Otsu and adaptive thresholds: Ch. 11.1 and 11.2</li>
 *   <li>opening and closing: Ch. 9.2 and 9.3</li>
 *   <li>contours, area, bounding box, circularity, convex hull, moments, eccentricity: Ch. 10.1 to 10.6</li>
 *   <li>circle Hough transform: Ch. 8.5 (OpenCV implements the gradient variant)</li>
 *   <li>scale-space blob detection and edge-response rejection: Ch. 25.1 and 25.2</li>
 * </ul>
 * OpenCV's versions of these algorithms differ in detail from the book's.
 * Intersection-over-union overlap suppression and the row-based head size are not textbook algorithms.
 */
public class HeadDetector {

    /**
     * Detection strategy.
     * <ul>
     *   <li>{@link #THRESHOLD}: dark regions from a binary threshold, then contours</li>
     *   <li>{@link #EDGES}: Canny edge outlines, filled, then contours</li>
     *   <li>{@link #BOTH}: union of the threshold mask and the edge mask</li>
     *   <li>{@link #HOUGH}: circle Hough transform; every detected circle is one box</li>
     *   <li>{@link #BLOB}: scale-space blob detection with row-dependent head size</li>
     * </ul>
     */
    public enum Method { THRESHOLD, EDGES, BOTH, HOUGH, BLOB }
    /**
     * How the grayscale image is binarised for {@link Method#THRESHOLD} and {@link Method#BOTH}.
     * Dark pixels become foreground because hair is darker than its surroundings.
     * <ul>
     *   <li>{@link #OTSU}: automatic global threshold from the histogram (Ch. 11.1.4)</li>
     *   <li>{@link #ADAPTIVE}: threshold from the local mean, copes with uneven lighting (Ch. 11.2)</li>
     *   <li>{@link #GLOBAL}: one fixed grey value for the whole image (Ch. 11.1)</li>
     * </ul>
     */
    public enum Threshold { OTSU, ADAPTIVE, GLOBAL }

    /**
     * All tunable parameters in one mutable bag with public fields. The defaults were found offline by
     * {@code com.classcounter.tools.ParamTuner}; the GUI sliders edit these fields directly.
     */
    public static class Params {
        // Defaults found by tools.ParamTuner on SCUT-HEAD Part B (objective "mix", 120 tuning / 60 held-out images):
        // held-out F1 = 0.074, mean |count error| = 51.5 %  (see results/RESULTS.md)
        public Method method = Method.THRESHOLD;
        public Threshold threshold = Threshold.GLOBAL;
        public int blur = 1;            // odd Gaussian kernel
        public int globalValue = 35;    // used by GLOBAL
        public int adaptiveBlock = 79;  // odd
        public int adaptiveC = 12;
        public int cannyLow = 35, cannyHigh = 185;
        public int morph = 7;           // structuring element size
        public double minArea = 223, maxArea = 5755;
        public double minAspect = 0.42, maxAspect = 1.84;
        public double minCircularity = 0.00;
        public double iou = 0.22;
        // Hough circle method (textbook Ch. 8.5; OpenCV uses the gradient variant)
        public int houghParam2 = 26;    // accumulator threshold: lower finds more circles
        public int houghMinR = 5, houghMaxR = 30;
        public int houghMinDist = 13;   // minimum distance between circle centres
        // extra shape filters (off by default): Ch. 10.4.2 convex hull, Ch. 10.6.2 eccentricity
        public double minSolidity = 0;       // area / convex-hull area, 0..1
        public double maxEccentricity = 9.70;  // major-axis / minor-axis ratio, 1 = circle
        // scale-space blob method (BLOB): head radius shrinks from the bottom (front rows) to the top (back rows)
        public int blobThreshold = 35;       // minimum blob contrast (scale-normalised LoG response)
        public int blobRTop = 6, blobRBottom = 18;   // expected head radius in px at the top / bottom image row
        public double blobTol = 0.6;         // accepted deviation from the expected radius (0.6 = +-60 %)
    }

    /**
     * One detection: a bounding box plus the measurements that produced it.
     *
     * @param rect        bounding box in pixels of the (possibly downscaled) analysed image
     * @param area        region area in square pixels (circle area for Hough and blob boxes)
     * @param circularity 4*pi*area / perimeter^2, 1.0 for a perfect circle
     */
    public record Box(Rect rect, double area, double circularity) {
        /**
         * Intersection over union of two boxes: overlap area divided by combined area.
         *
         * @param o the other box
         * @return a value from 0 (disjoint or touching) to 1 (identical)
         */
        double iou(Box o) {
            Rect a = rect, b = o.rect;
            int x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
            int x2 = Math.min(a.x + a.width, b.x + b.width), y2 = Math.min(a.y + a.height, b.y + b.height);
            double inter = Math.max(0, x2 - x1) * (double) Math.max(0, y2 - y1);
            double union = a.area() + b.area() - inter;
            return union <= 0 ? 0 : inter / union;
        }
    }

    /** Output of one run: boxes plus intermediate images for the stage viewer. */
    public record Result(List<Box> boxes, int rawContours, Mat gray, Mat blurred, Mat edges, Mat binary, Mat cleaned) {}

    /**
     * Runs the full pipeline on one image.
     *
     * @param bgr colour image in OpenCV's BGR order
     * @param p   parameters, see {@link Params}
     * @return the surviving boxes plus the intermediate images for display
     */
    public Result detect(Mat bgr, Params p) {
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(odd(p.blur), odd(p.blur)), 0);

        if (p.method == Method.BLOB) return blob(gray, blurred, p);

        Mat edges = new Mat();
        Imgproc.Canny(blurred, edges, p.cannyLow, Math.max(p.cannyLow, p.cannyHigh));

        // heads (dark hair) are foreground -> inverted binary
        Mat binary = new Mat();
        switch (p.threshold) {
            case OTSU -> Imgproc.threshold(blurred, binary, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);
            case GLOBAL -> Imgproc.threshold(blurred, binary, p.globalValue, 255, Imgproc.THRESH_BINARY_INV);
            case ADAPTIVE -> Imgproc.adaptiveThreshold(blurred, binary, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                    Imgproc.THRESH_BINARY_INV, odd(Math.max(3, p.adaptiveBlock)), p.adaptiveC);
        }

        Mat k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(p.morph, p.morph));
        Mat mask = new Mat();
        switch (p.method) {
            case THRESHOLD, HOUGH -> binary.copyTo(mask);
            case EDGES -> Imgproc.dilate(edges, mask, k);
            case BOTH -> { Mat e = new Mat(); Imgproc.dilate(edges, e, k); Core.bitwise_or(binary, e, mask); }
        }
        Mat cleaned = new Mat();
        Imgproc.morphologyEx(mask, cleaned, Imgproc.MORPH_OPEN, k);
        Imgproc.morphologyEx(cleaned, cleaned, Imgproc.MORPH_CLOSE, k);
        if (p.method == Method.EDGES || p.method == Method.BOTH) fill(cleaned);
        if (p.method == Method.HOUGH) return hough(gray, blurred, edges, binary, cleaned, p, bgr.size());

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(cleaned.clone(), contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Box> kept = new ArrayList<>();
        // thousands of contours means the settings produced noise; skip the costly per-contour work
        if (contours.size() > 4000) return new Result(kept, contours.size(), gray, blurred, edges, binary, cleaned);
        for (MatOfPoint c : contours) {
            double area = Imgproc.contourArea(c);
            if (area < p.minArea || area > p.maxArea) continue;
            Rect r = Imgproc.boundingRect(c);
            double aspect = r.width / (double) r.height;
            if (aspect < p.minAspect || aspect > p.maxAspect) continue;
            MatOfPoint2f c2 = new MatOfPoint2f(c.toArray());
            double per = Imgproc.arcLength(c2, true);
            double circ = per == 0 ? 0 : 4 * Math.PI * area / (per * per);
            if (circ < p.minCircularity) continue;
            if (p.minSolidity > 0 && solidity(c, area) < p.minSolidity) continue;
            if (p.maxEccentricity < 10 && eccentricity(c) > p.maxEccentricity) continue;
            kept.add(new Box(r, area, circ));
        }
        return new Result(suppress(kept, p.iou), contours.size(), gray, blurred, edges, binary, cleaned);
    }

    private static final int MAX_BLOB_PEAKS = 3000;   // more raw peaks than this means noise

    /**
     * A box with the blob response that produced it, used to let the strongest blob win an overlap.
     */
    private record Scored(Box box, double score) {}

    /**
     * Scale-space blob detection (idea of book Ch. 25.1, interest points at multiple scales).
     * Dark blobs give strong peaks of the scale-normalised Laplacian of Gaussian (sigma^2 * LoG);
     * the sigma of the peak gives the blob radius (r = sigma * sqrt 2). Because every head is its own
     * peak, touching heads stay separate. The expected head radius is interpolated linearly from the
     * top row (back of the room) to the bottom row (front), and blobs far from it are rejected.
     */
    private Result blob(Mat gray, Mat blurred, Params p) {
        int w = gray.cols(), h = gray.rows(), n = 8;
        double rTop = Math.max(2, p.blobRTop), rBot = Math.max(2, p.blobRBottom);
        double rLo = Math.max(2, Math.min(rTop, rBot) * (1 - p.blobTol));
        double rHi = Math.max(rLo * 1.2, Math.max(rTop, rBot) * (1 + p.blobTol));
        Mat g32 = new Mat();
        gray.convertTo(g32, CvType.CV_32F);

        double[] sigma = new double[n];
        Mat[] resp = new Mat[n], dil = new Mat[n], ok = new Mat[n];
        Mat k3 = Mat.ones(3, 3, CvType.CV_8U);
        Mat best = Mat.zeros(h, w, CvType.CV_32F);
        for (int i = 0; i < n; i++) {
            sigma[i] = rLo * Math.pow(rHi / rLo, i / (n - 1.0)) / Math.sqrt(2);
            Mat b = new Mat(), lap = new Mat();
            Imgproc.GaussianBlur(g32, b, new Size(), sigma[i]);
            Imgproc.Laplacian(b, lap, CvType.CV_32F, 1);
            Core.multiply(lap, new Scalar(sigma[i] * sigma[i]), lap);   // scale normalisation
            resp[i] = lap;
            ok[i] = notEdge(b);   // book Ch. 25.2: eliminate edge (ridge) responses
            dil[i] = new Mat();
            Imgproc.dilate(lap, dil[i], k3);
            Core.max(best, lap, best);
            b.release();
        }

        List<Scored> found = new ArrayList<>();
        int rawTotal = 0;
        boolean noisy = false;   // settings so sensitive that noise fires everywhere: give up early
        for (int i = 0; i < n; i++) {
            Mat m = dil[i].clone();
            if (i > 0) Core.max(m, dil[i - 1], m);
            if (i < n - 1) Core.max(m, dil[i + 1], m);
            Mat isMax = new Mat(), above = new Mat(), peaks = new Mat(), locs = new Mat();
            Core.compare(resp[i], m, isMax, Core.CMP_GE);
            Core.compare(resp[i], new Scalar(p.blobThreshold), above, Core.CMP_GT);
            Core.bitwise_and(isMax, above, peaks);
            Core.bitwise_and(peaks, ok[i], peaks);
            Core.findNonZero(peaks, locs);
            if (locs.rows() > MAX_BLOB_PEAKS || found.size() > MAX_BLOB_PEAKS) {   // noise: settings far too sensitive
                for (int q = 0; q < n; q++) { resp[q].release(); dil[q].release(); ok[q].release(); }
                Mat zero = Mat.zeros(h, w, CvType.CV_8UC1);
                return new Result(new ArrayList<>(), MAX_BLOB_PEAKS + 1, gray, blurred, zero, zero.clone(), zero.clone());
            }
            rawTotal += locs.rows();
            if (rawTotal > 40000) noisy = true;
            for (int j = 0; !noisy && j < locs.rows(); j++) {
                if (found.size() > 4000) { noisy = true; break; }
                double[] pt = locs.get(j, 0);
                double x = pt[0], y = pt[1], r = sigma[i] * Math.sqrt(2);
                double expected = rTop + (rBot - rTop) * y / Math.max(1, h - 1);
                if (r < expected * (1 - p.blobTol) || r > expected * (1 + p.blobTol)) continue;
                int bx = (int) Math.max(0, x - r), by = (int) Math.max(0, y - r);
                int bw = (int) Math.min(w - bx, 2 * r), bh = (int) Math.min(h - by, 2 * r);
                if (bw > 0 && bh > 0)
                    found.add(new Scored(new Box(new Rect(bx, by, bw, bh), Math.PI * r * r, 1.0), resp[i].get((int) y, (int) x)[0]));
            }
            m.release(); isMax.release(); above.release(); peaks.release(); locs.release();
            if (noisy) break;
        }
        if (noisy) {
            for (int i = 0; i < n; i++) { resp[i].release(); dil[i].release(); ok[i].release(); }
            Mat zero = Mat.zeros(h, w, CvType.CV_8UC1);
            return new Result(new ArrayList<>(), Math.max(rawTotal, found.size()), gray, blurred, zero, zero.clone(), zero.clone());
        }

        found.sort(Comparator.comparingDouble(Scored::score).reversed());   // strongest blob wins an overlap
        List<Box> out = new ArrayList<>();
        for (Scored sc : found) {
            boolean dup = false;
            for (Box k : out) if (k.iou(sc.box()) > p.iou) { dup = true; break; }
            if (!dup) out.add(sc.box());
        }

        Mat view = new Mat();
        Core.max(best, Mat.zeros(h, w, CvType.CV_32F), best);
        Core.normalize(best, view, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);   // strongest blob response, for the stage viewer
        for (int i = 0; i < n; i++) { resp[i].release(); dil[i].release(); ok[i].release(); }
        Mat zero = Mat.zeros(h, w, CvType.CV_8UC1);
        return new Result(out, found.size(), gray, blurred, zero, zero.clone(), view);
    }

    /**
     * Edge-response rejection as in SIFT key point selection (book Ch. 25.2): a ridge or bar has one large and
     * one small principal curvature, a round blob has two similar ones. Keeps pixels where
     * trace^2 / det < (r + 1)^2 / r with r = 10 (the Hessian entries come from Sobel derivatives).
     */
    private static Mat notEdge(Mat blurred) {
        final double r = 10;
        Mat dxx = new Mat(), dyy = new Mat(), dxy = new Mat();
        Imgproc.Sobel(blurred, dxx, CvType.CV_32F, 2, 0, 3);
        Imgproc.Sobel(blurred, dyy, CvType.CV_32F, 0, 2, 3);
        Imgproc.Sobel(blurred, dxy, CvType.CV_32F, 1, 1, 3);
        Mat det = new Mat(), t = new Mat(), tr = new Mat();
        Core.multiply(dxx, dyy, det);
        Core.multiply(dxy, dxy, t);
        Core.subtract(det, t, det);                      // det = dxx*dyy - dxy^2
        Core.add(dxx, dyy, tr);
        Core.multiply(tr, tr, tr);                       // trace^2
        Core.multiply(tr, new Scalar(r), tr);            // r * trace^2
        Core.multiply(det, new Scalar((r + 1) * (r + 1)), det);
        Mat positive = new Mat(), curvOk = new Mat(), ok = new Mat();
        Core.compare(det, new Scalar(0), positive, Core.CMP_GT);
        Core.compare(tr, det, curvOk, Core.CMP_LT);      // r*tr^2 < (r+1)^2 * det_original
        Core.bitwise_and(positive, curvOk, ok);
        dxx.release(); dyy.release(); dxy.release(); det.release(); t.release(); tr.release(); positive.release(); curvOk.release();
        return ok;
    }

    /** Solidity = region area / convex hull area (book Ch. 10.4.2); low for ragged or merged shapes. */
    static double solidity(MatOfPoint contour, double area) {
        MatOfInt hullIdx = new MatOfInt();
        Imgproc.convexHull(contour, hullIdx);
        Point[] pts = contour.toArray();
        int[] idx = hullIdx.toArray();
        Point[] hp = new Point[idx.length];
        for (int i = 0; i < idx.length; i++) hp[i] = pts[idx[i]];
        double hullArea = Imgproc.contourArea(new MatOfPoint(hp));
        return hullArea > 0 ? Math.min(1.0, area / hullArea) : 0;
    }

    /**
     * Eccentricity from central moments (book Ch. 10.6.2): ratio of the major to the minor axis
     * of the equivalent ellipse. 1 = circle, large = elongated.
     */
    static double eccentricity(MatOfPoint contour) {
        Moments m = Imgproc.moments(contour);
        double root = Math.sqrt((m.mu20 - m.mu02) * (m.mu20 - m.mu02) + 4 * m.mu11 * m.mu11);
        double a1 = m.mu20 + m.mu02 + root, a2 = m.mu20 + m.mu02 - root;
        return a2 <= 1e-9 ? Double.MAX_VALUE : a1 / a2;
    }

    /** Circle Hough transform: each detected circle becomes one box (splits touching round heads). */
    private Result hough(Mat gray, Mat blurred, Mat edges, Mat binary, Mat cleaned, Params p, Size size) {
        Mat circles = new Mat();
        int minR = Math.max(1, p.houghMinR);
        Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.2, Math.max(1, p.houghMinDist),
                Math.max(10, p.cannyHigh), Math.max(5, p.houghParam2), minR, Math.max(minR + 1, p.houghMaxR));
        List<Box> boxes = new ArrayList<>();
        for (int i = 0; i < circles.cols(); i++) {
            double[] c = circles.get(0, i);
            int r = (int) Math.round(c[2]);
            int x = (int) Math.max(0, c[0] - r), y = (int) Math.max(0, c[1] - r);
            int w = (int) Math.min(size.width - x, 2 * r), h = (int) Math.min(size.height - y, 2 * r);
            if (w > 0 && h > 0) boxes.add(new Box(new Rect(x, y, w, h), Math.PI * r * r, 1.0));
        }
        return new Result(suppress(boxes, p.iou), circles.cols(), gray, blurred, edges, binary, cleaned);
    }

    /** Fills closed outlines (e.g. Canny rings) so each head becomes a solid region. */
    private static void fill(Mat m) {
        List<MatOfPoint> cs = new ArrayList<>();
        Imgproc.findContours(m.clone(), cs, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(m, cs, -1, new Scalar(255), -1);
    }

    /** Greedy overlap suppression: keep the larger box when IoU exceeds the limit. */
    static List<Box> suppress(List<Box> boxes, double limit) {
        List<Box> sorted = new ArrayList<>(boxes);
        sorted.sort(Comparator.comparingDouble(Box::area).reversed());
        List<Box> out = new ArrayList<>();
        for (Box b : sorted) {
            boolean dup = false;
            for (Box k : out) if (k.iou(b) > limit) { dup = true; break; }
            if (!dup) out.add(b);
        }
        return out;
    }

    /**
     * Forces a kernel size to be a positive odd number.
     */
    private static int odd(int n) { n = Math.max(1, n); return n % 2 == 0 ? n + 1 : n; }

    /** Draws boxes and index labels on a copy of the image. */
    public static Mat annotate(Mat bgr, List<Box> boxes) {
        Mat out = bgr.clone();
        int i = 1;
        for (Box b : boxes) {
            Imgproc.rectangle(out, b.rect().tl(), b.rect().br(), new Scalar(0, 220, 0), 2);
            Imgproc.putText(out, String.valueOf(i++), new Point(b.rect().x, Math.max(10, b.rect().y - 3)),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, new Scalar(0, 255, 255), 1);
        }
        return out;
    }
}
