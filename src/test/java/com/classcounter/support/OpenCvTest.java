package com.classcounter.support;

import com.classcounter.util.NativeLoader;
import org.junit.jupiter.api.BeforeAll;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/** Base class for tests that need OpenCV: loads the native library once and builds synthetic images. */
public abstract class OpenCvTest {

    @BeforeAll
    static void loadOpenCv() {
        NativeLoader.load();
    }

    /** Light grey colour image with dark filled discs of the given radius at the given centres. */
    protected static Mat discs(int width, int height, int radius, int darkness, Point... centres) {
        Mat image = new Mat(height, width, CvType.CV_8UC3, new Scalar(190, 190, 190));
        for (Point c : centres) {
            Imgproc.circle(image, c, radius, new Scalar(darkness, darkness, darkness), -1);
        }
        return image;
    }

    /** Light grey colour image with one dark filled rectangle. */
    protected static Mat bar(int width, int height, int x, int y, int w, int h, int darkness) {
        Mat image = new Mat(height, width, CvType.CV_8UC3, new Scalar(190, 190, 190));
        Imgproc.rectangle(image, new Point(x, y), new Point(x + w, y + h), new Scalar(darkness, darkness, darkness), -1);
        return image;
    }
}
