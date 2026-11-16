package com.classcounter.tools;

import com.classcounter.support.OpenCvTest;
import com.classcounter.vision.HeadDetector.Method;
import com.classcounter.vision.HeadDetector.Params;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Tests the offline tuner: scoring, search moves, output format and VOC dataset loading. */
class ParamTunerTest extends OpenCvTest {

    @AfterEach
    void resetStaticState() {
        ParamTuner.forced = null;
        ParamTuner.mode = "f1";
        ParamTuner.matchIou = 0.5;
        ParamTuner.minGtArea = 100;
    }

    // ---------- scoring ----------

    @Test
    void scoreComputesPrecisionRecallAndF1() {
        ParamTuner.Score s = new ParamTuner.Score(10, 10, 30, 0.2);
        assertEquals(0.5, s.precision(), 1e-9);
        assertEquals(0.25, s.recall(), 1e-9);
        assertEquals(1.0 / 3.0, s.f1(), 1e-9);
    }

    @Test
    void scoreWithoutAnyDetectionHasZeroF1() {
        assertEquals(0, new ParamTuner.Score(0, 0, 5, 1.0).f1(), 1e-9);
    }

    @Test
    void mixObjectiveNeedsBothGoodBoxesAndAGoodTotal() {
        ParamTuner.mode = "mix";
        assertEquals(0.5 * (1 - 0.2), new ParamTuner.Score(10, 10, 10, 0.2).objective(), 1e-9);
        assertEquals(0, new ParamTuner.Score(10, 10, 10, 1.5).objective(), 1e-9, "a huge count error scores zero");
    }

    @Test
    void countObjectivePrefersSmallerCountError() {
        ParamTuner.mode = "count";
        assertTrue(new ParamTuner.Score(1, 1, 1, 0.1).objective() > new ParamTuner.Score(1, 1, 1, 0.9).objective());
    }

    @Test
    void iouOfRectangles() {
        assertEquals(1.0, ParamTuner.iou(new Rect(0, 0, 10, 10), new Rect(0, 0, 10, 10)), 1e-9);
        assertEquals(0.0, ParamTuner.iou(new Rect(0, 0, 10, 10), new Rect(20, 20, 5, 5)), 1e-9);
        assertEquals(1.0 / 3.0, ParamTuner.iou(new Rect(0, 0, 10, 10), new Rect(5, 0, 10, 10)), 1e-9);
    }

    @Test
    void evalFindsAllDiscsOfASyntheticImage() {
        Point[] centres = {new Point(80, 100), new Point(200, 100), new Point(320, 100)};
        Mat image = discs(400, 200, 15, 10, centres);
        List<Rect> gt = List.of(
                new Rect(65, 85, 31, 31), new Rect(185, 85, 31, 31), new Rect(305, 85, 31, 31));
        ParamTuner.Score s = ParamTuner.eval(List.of(new ParamTuner.Sample("synthetic", image, gt)), new Params());
        assertEquals(3, s.tp());
        assertEquals(0, s.fp());
        assertEquals(0, s.fn());
        assertEquals(0, s.countErr(), 1e-9);
    }

    @Test
    void matchIouControlsWhatCountsAsAHit() {
        Mat image = discs(400, 200, 15, 10, new Point(200, 100));
        List<Rect> shifted = List.of(new Rect(195, 85, 31, 31));   // 5 px off: IoU about 0.7
        ParamTuner.Sample sample = new ParamTuner.Sample("shifted", image, shifted);
        ParamTuner.matchIou = 0.5;
        assertEquals(1, ParamTuner.eval(List.of(sample), new Params()).tp());
        ParamTuner.matchIou = 0.9;
        assertEquals(0, ParamTuner.eval(List.of(sample), new Params()).tp());
    }

    // ---------- search moves ----------

    @Test
    void mutationAlwaysStaysInsideValidRanges() {
        Random rnd = new Random(7);
        Params p = new Params();
        for (int i = 0; i < 2000; i++) {
            p = ParamTuner.mutate(p, rnd);
            assertTrue(p.blur >= 1 && p.blur <= 15);
            assertTrue(p.cannyHigh >= p.cannyLow);
            assertTrue(p.minArea >= 10 && p.maxArea > p.minArea);
            assertTrue(p.minAspect < p.maxAspect);
            assertTrue(p.minCircularity >= 0 && p.minCircularity <= 0.9);
            assertTrue(p.iou >= 0.05 && p.iou <= 0.8);
            assertTrue(p.minSolidity >= 0 && p.minSolidity <= 0.95);
            assertTrue(p.maxEccentricity >= 1.3 && p.maxEccentricity <= 10);
            assertTrue(p.houghMaxR >= p.houghMinR + 2);
            assertTrue(p.blobThreshold >= 10 && p.blobRTop >= 2 && p.blobRBottom >= 2);
        }
    }

    @Test
    void mutationDoesNotChangeTheOriginal() {
        Params original = new Params();
        double before = original.minArea;
        for (int i = 0; i < 50; i++) ParamTuner.mutate(original, new Random(i));
        assertEquals(before, original.minArea, 0);
    }

    @Test
    void forcedMethodIsRespectedByRandomSampling() {
        ParamTuner.forced = Method.BLOB;
        Random rnd = new Random(3);
        for (int i = 0; i < 100; i++) assertEquals(Method.BLOB, ParamTuner.random(rnd).method);
    }

    @Test
    void randomSamplingReachesEveryMethodWhenNotForced() {
        Random rnd = new Random(3);
        java.util.Set<Method> seen = new java.util.HashSet<>();
        for (int i = 0; i < 300; i++) seen.add(ParamTuner.random(rnd).method);
        assertEquals(Method.values().length, seen.size());
    }

    @Test
    void randomBlobSettingsDoNotStartInTheNoisyRange() {
        Random rnd = new Random(5);
        for (int i = 0; i < 300; i++) assertTrue(ParamTuner.random(rnd).blobThreshold >= 15);
    }

    @Test
    void copyIsIndependentOfTheOriginal() {
        Params a = new Params();
        Params b = ParamTuner.copy(a);
        b.blobTol = 1.4;
        assertNotEquals(a.blobTol, b.blobTol, 0);
    }

    // ---------- output and arguments ----------

    @Test
    void javaOutputListsEveryParameterWithDotDecimals() {
        String out = ParamTuner.asJava(new Params());
        for (String field : new String[]{"method", "threshold", "blur", "globalValue", "adaptiveBlock", "adaptiveC",
                "cannyLow", "morph", "minArea", "minAspect", "minCircularity", "iou", "houghParam2",
                "minSolidity", "blobThreshold", "blobTol"}) {
            assertTrue(out.contains(field), "missing " + field);
        }
        assertFalse(out.matches("(?s).*\\d,\\d\\d.*"), "decimals must use a dot");
    }

    @Test
    void argumentsAreParsedInPairs() {
        Map<String, String> m = ParamTuner.parse(new String[]{"--root", "data", "--n", "40"});
        assertEquals("data", m.get("root"));
        assertEquals("40", m.get("n"));
    }

    @Test
    void clampLimitsValues() {
        assertEquals(5, ParamTuner.clamp(1, 5, 10));
        assertEquals(10, ParamTuner.clamp(99, 5, 10));
        assertEquals(7, ParamTuner.clamp(7, 5, 10));
    }

    // ---------- dataset loading (Pascal VOC layout) ----------

    private static void writeVoc(Path root, String name, int w, int h, int... boxes) throws Exception {
        Files.createDirectories(root.resolve("JPEGImages"));
        Files.createDirectories(root.resolve("Annotations"));
        Mat image = new Mat(h, w, org.opencv.core.CvType.CV_8UC3, new org.opencv.core.Scalar(190, 190, 190));
        Imgcodecs.imwrite(root.resolve("JPEGImages/" + name + ".jpg").toString(), image);
        StringBuilder xml = new StringBuilder("<annotation>");
        for (int i = 0; i < boxes.length; i += 4) {
            xml.append("<object><name>head</name><bndbox><xmin>").append(boxes[i]).append("</xmin><ymin>")
               .append(boxes[i + 1]).append("</ymin><xmax>").append(boxes[i + 2]).append("</xmax><ymax>")
               .append(boxes[i + 3]).append("</ymax></bndbox></object>");
        }
        Files.writeString(root.resolve("Annotations/" + name + ".xml"), xml.append("</annotation>").toString());
    }

    @Test
    void loadsImagesAndBoxesFromAVocFolder(@TempDir Path root) throws Exception {
        writeVoc(root, "a", 200, 100, 10, 10, 50, 50, 100, 20, 140, 60);
        List<String> names = ParamTuner.listNames(root, "train");
        assertEquals(List.of("a"), names);
        List<ParamTuner.Sample> samples = ParamTuner.load(root, names);
        assertEquals(1, samples.size());
        assertEquals(2, samples.get(0).gt().size());
    }

    @Test
    void usesTheSplitListWhenPresent(@TempDir Path root) throws Exception {
        writeVoc(root, "a", 100, 100, 10, 10, 50, 50);
        writeVoc(root, "b", 100, 100, 10, 10, 50, 50);
        Files.createDirectories(root.resolve("ImageSets/Main"));
        Files.writeString(root.resolve("ImageSets/Main/train.txt"), "b\n");
        assertEquals(List.of("b"), ParamTuner.listNames(root, "train"));
    }

    @Test
    void ignoresGroundTruthBoxesBelowTheMinimumArea(@TempDir Path root) throws Exception {
        writeVoc(root, "a", 200, 100, 10, 10, 14, 14, 50, 50, 90, 90);   // 4x4 box and 40x40 box
        ParamTuner.minGtArea = 100;
        assertEquals(1, ParamTuner.load(root, List.of("a")).get(0).gt().size());
    }

    @Test
    void shrinksWideImagesAndScalesTheBoxesTheSameWay(@TempDir Path root) throws Exception {
        writeVoc(root, "wide", 2560, 1440, 1000, 400, 1200, 600);        // 200 x 200 px at full size
        ParamTuner.minGtArea = 1;
        ParamTuner.Sample s = ParamTuner.load(root, List.of("wide")).get(0);
        assertEquals(1280, s.image().cols());
        Rect r = s.gt().get(0);
        assertEquals(500, r.x);
        assertEquals(100, r.width);
    }
}
