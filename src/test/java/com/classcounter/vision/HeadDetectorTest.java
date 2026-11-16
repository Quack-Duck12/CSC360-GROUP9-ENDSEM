package com.classcounter.vision;

import com.classcounter.support.OpenCvTest;
import com.classcounter.vision.HeadDetector.Box;
import com.classcounter.vision.HeadDetector.Method;
import com.classcounter.vision.HeadDetector.Params;
import com.classcounter.vision.HeadDetector.Result;
import org.junit.jupiter.api.Test;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Tests the detection pipeline on small synthetic images where the right answer is known. */
class HeadDetectorTest extends OpenCvTest {

    private final HeadDetector detector = new HeadDetector();

    private static Box box(int x, int y, int w, int h) {
        return new Box(new Rect(x, y, w, h), w * (double) h, 1.0);
    }

    private static MatOfPoint outlineOf(Mat mask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask.clone(), contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours.get(0);
    }

    // ---------- threshold method ----------

    @Test
    void thresholdMethodCountsEachDarkDisc() {
        Mat image = discs(400, 200, 15, 10, new Point(80, 100), new Point(200, 100), new Point(320, 100));
        Params p = new Params();   // tuned defaults: THRESHOLD + GLOBAL 35
        assertEquals(3, detector.detect(image, p).boxes().size());
    }

    @Test
    void regionsBelowTheMinimumAreaAreRejected() {
        Mat image = discs(200, 100, 4, 10, new Point(100, 50));   // area about 50 px^2
        assertEquals(0, detector.detect(image, new Params()).boxes().size());
    }

    @Test
    void wideBarFailsTheAspectRatioLimit() {
        Mat image = bar(400, 200, 100, 90, 200, 20, 10);          // 10:1
        assertEquals(0, detector.detect(image, new Params()).boxes().size());
    }

    @Test
    void blankImageGivesNoBoxesForEveryMethod() {
        Mat blank = new Mat(120, 160, CvType.CV_8UC3, new Scalar(190, 190, 190));
        for (Method m : Method.values()) {
            Params p = new Params();
            p.method = m;
            assertEquals(0, detector.detect(blank, p).boxes().size(), "method " + m);
        }
    }

    @Test
    void stageImagesHaveTheSizeOfTheInput() {
        Mat image = discs(300, 150, 15, 10, new Point(150, 75));
        Result r = detector.detect(image, new Params());
        assertEquals(image.size(), r.gray().size());
        assertEquals(image.size(), r.blurred().size());
        assertEquals(image.size(), r.edges().size());
        assertEquals(image.size(), r.binary().size());
        assertEquals(image.size(), r.cleaned().size());
    }

    @Test
    void evenGaussianKernelIsAcceptedAndMadeOdd() {
        Mat image = discs(300, 150, 15, 10, new Point(150, 75));
        Params p = new Params();
        p.blur = 4;
        assertEquals(1, detector.detect(image, p).boxes().size());
    }

    @Test
    void otsuAndAdaptiveAlsoFindADarkDisc() {
        Mat image = discs(300, 150, 15, 10, new Point(150, 75));
        for (HeadDetector.Threshold t : new HeadDetector.Threshold[]{HeadDetector.Threshold.OTSU, HeadDetector.Threshold.ADAPTIVE}) {
            Params p = new Params();
            p.threshold = t;
            p.blur = 5;
            p.adaptiveBlock = 51;
            assertTrue(detector.detect(image, p).boxes().size() >= 1, "threshold " + t);
        }
    }

    // ---------- overlap suppression ----------

    @Test
    void heavilyOverlappingBoxesKeepOnlyTheLarger() {
        Box large = box(0, 0, 30, 30), small = box(2, 2, 28, 28);
        List<Box> kept = HeadDetector.suppress(List.of(small, large), 0.3);
        assertEquals(1, kept.size());
        assertSame(large, kept.get(0));
    }

    @Test
    void disjointBoxesAreAllKept() {
        List<Box> kept = HeadDetector.suppress(List.of(box(0, 0, 10, 10), box(50, 0, 10, 10), box(0, 50, 10, 10)), 0.3);
        assertEquals(3, kept.size());
    }

    @Test
    void aStricterOverlapLimitRemovesMoreBoxes() {
        List<Box> boxes = List.of(box(0, 0, 10, 10), box(8, 0, 10, 10));   // IoU 0.11
        assertEquals(2, HeadDetector.suppress(boxes, 0.3).size());
        assertEquals(1, HeadDetector.suppress(boxes, 0.05).size());
    }

    @Test
    void suppressionHandlesEmptyInput() {
        assertTrue(HeadDetector.suppress(List.of(), 0.3).isEmpty());
    }

    @Test
    void boxIouMatchesHandCalculation() {
        assertEquals(1.0, box(5, 5, 10, 10).iou(box(5, 5, 10, 10)), 1e-9);
        assertEquals(0.0, box(0, 0, 10, 10).iou(box(20, 20, 5, 5)), 1e-9);
        assertEquals(0.0, box(0, 0, 10, 10).iou(box(10, 0, 10, 10)), 1e-9);          // touching only
        assertEquals(1.0 / 3.0, box(0, 0, 10, 10).iou(box(5, 0, 10, 10)), 1e-9);     // 50 / 150
    }

    // ---------- shape features (book Ch. 10.4 and 10.6) ----------

    @Test
    void solidOfASquareIsOneAndAnLShapeIsLower() {
        Mat square = Mat.zeros(60, 60, CvType.CV_8UC1);
        Imgproc.rectangle(square, new Point(10, 10), new Point(49, 49), new Scalar(255), -1);
        MatOfPoint sq = outlineOf(square);
        assertEquals(1.0, HeadDetector.solidity(sq, Imgproc.contourArea(sq)), 0.02);

        Mat ell = Mat.zeros(60, 60, CvType.CV_8UC1);
        Imgproc.rectangle(ell, new Point(10, 10), new Point(49, 24), new Scalar(255), -1);
        Imgproc.rectangle(ell, new Point(10, 10), new Point(24, 49), new Scalar(255), -1);
        MatOfPoint l = outlineOf(ell);
        assertTrue(HeadDetector.solidity(l, Imgproc.contourArea(l)) < 0.8);
    }

    @Test
    void eccentricityIsOneForACircleAndLargeForABar() {
        Mat circle = Mat.zeros(80, 80, CvType.CV_8UC1);
        Imgproc.circle(circle, new Point(40, 40), 25, new Scalar(255), -1);
        assertEquals(1.0, HeadDetector.eccentricity(outlineOf(circle)), 0.2);

        Mat bar = Mat.zeros(80, 200, CvType.CV_8UC1);
        Imgproc.rectangle(bar, new Point(20, 30), new Point(179, 69), new Scalar(255), -1);   // about 4:1
        assertTrue(HeadDetector.eccentricity(outlineOf(bar)) > 10);
    }

    @Test
    void eccentricityFilterRemovesElongatedRegions() {
        Mat image = bar(400, 200, 100, 80, 140, 30, 10);          // about 4.7:1 bar, aspect limit disabled below
        Params p = new Params();
        p.minAspect = 0.05;
        p.maxAspect = 20;
        p.maxEccentricity = 10;   // 10 means "off" (the tuned default of 9.7 is already active)
        assertEquals(1, detector.detect(image, p).boxes().size());
        p.maxEccentricity = 3.0;
        assertEquals(0, detector.detect(image, p).boxes().size());
    }

    // ---------- Hough method ----------

    @Test
    void houghMethodFindsSeparatedCircles() {
        Mat image = discs(400, 200, 18, 30, new Point(80, 100), new Point(200, 100), new Point(320, 100));
        Params p = new Params();
        p.method = Method.HOUGH;
        p.blur = 5;
        assertEquals(3, detector.detect(image, p).boxes().size());
    }

    // ---------- blob method (book Ch. 25) ----------

    @Test
    void blobMethodFindsSeparatedDiscsOfTheExpectedSize() {
        Mat image = discs(400, 300, 12, 40, new Point(80, 150), new Point(200, 150), new Point(320, 150));
        Params p = new Params();
        p.method = Method.BLOB;
        assertEquals(3, detector.detect(image, p).boxes().size());
    }

    @Test
    void blobMethodKeepsTwoTouchingDiscsApart() {
        Mat image = discs(400, 300, 12, 40, new Point(190, 150), new Point(212, 150));
        Params p = new Params();
        p.method = Method.BLOB;
        assertEquals(2, detector.detect(image, p).boxes().size());
        p.method = Method.THRESHOLD;
        p.globalValue = 100;
        assertEquals(1, detector.detect(image, p).boxes().size(), "the threshold method merges them");
    }

    @Test
    void blobMethodNeverReportsAWrongSizedDiscAsOneBigHead() {
        // at the top row heads are expected to be about 6 px in radius; a 25 px disc there is far too large.
        // Known limitation: its curved boundary can still fire a few small blobs, but no box may be head sized for
        // the disc itself (diameter 50).
        Mat image = discs(400, 300, 25, 40, new Point(200, 30));
        Params p = new Params();
        p.method = Method.BLOB;
        for (Box b : detector.detect(image, p).boxes()) {
            assertTrue(b.rect().width < 25, "box too large for this row: " + b.rect());
        }
    }

    @Test
    void blobEdgeRejectionSuppressesRidgeResponses() {
        Mat image = bar(400, 200, 100, 90, 200, 20, 10);          // a thin bar looks like a row of blobs without it
        Params p = new Params();
        p.method = Method.BLOB;
        assertTrue(detector.detect(image, p).boxes().size() <= 6, "only a few corner responses may remain");
    }

    @Test
    void blobMethodTreatsPureNoiseAsNoiseInsteadOfStalling() {
        Mat noise = new Mat(720, 1280, CvType.CV_8UC3);
        Core.randn(noise, 128, 40);
        Params p = new Params();
        p.method = Method.BLOB;
        p.blobThreshold = 3;   // absurdly sensitive
        long start = System.currentTimeMillis();
        Result r = detector.detect(noise, p);
        assertTrue(r.boxes().isEmpty());
        assertTrue(r.rawContours() > 3000);
        assertTrue(System.currentTimeMillis() - start < 20_000, "must return quickly");
    }

    // ---------- drawing ----------

    @Test
    void annotateDrawsOnACopyAndLeavesTheInputUntouched() {
        Mat image = new Mat(100, 100, CvType.CV_8UC3, new Scalar(0, 0, 0));
        Mat out = HeadDetector.annotate(image, List.of(box(10, 10, 40, 40)));
        assertEquals(image.size(), out.size());
        assertEquals(0, Core.countNonZero(image.reshape(1)), "input stays black");
        assertTrue(Core.countNonZero(out.reshape(1)) > 0, "output has the drawn box");
    }
}
