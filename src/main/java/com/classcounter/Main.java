package com.classcounter;

import com.classcounter.util.NativeLoader;
import com.classcounter.vision.HeadDetector;
import com.classcounter.vision.HeadDetector.Params;
import com.classcounter.vision.HeadDetector.Result;
import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.function.Consumer;

/** Minimal JavaFX front end: load an image, tune the pipeline, see the head count. */
public class Main extends Application {

    private static final int MAX_WIDTH = 1280;

    private final HeadDetector detector = new HeadDetector();
    private final Params p = new Params();
    private final ImageView view = new ImageView();
    private final Label status = new Label("Upload an image to count head candidates.");
    private final Label countNumber = new Label("-");
    private final ComboBox<String> stage = new ComboBox<>();
    private Mat image;       // BGR, downscaled
    private Result result;

    @Override
    /**
     * Builds the window: controls on the left, image view in the centre, status line at the bottom.
     */
    public void start(Stage primary) {
        NativeLoader.load();

        // ---- header bar: title, upload button, count ----
        Label title = new Label("Classroom Student Counter");
        title.setFont(Font.font("System", FontWeight.BOLD, 22));
        Label subtitle = new Label("Classical image processing");
        subtitle.setTextFill(Color.GRAY);
        VBox titleBox = new VBox(2, title, subtitle);

        Button upload = new Button("Upload Image");
        upload.setStyle("-fx-background-color: #2563eb; -fx-text-fill: white; -fx-font-weight: bold;");
        upload.setPadding(new Insets(8, 18, 8, 18));
        upload.setOnAction(e -> {
            FileChooser fc = new FileChooser();
            fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Images", "*.jpg", "*.jpeg", "*.png", "*.bmp"));
            File f = fc.showOpenDialog(primary);
            if (f != null) load(f);
        });

        Label countCaption = new Label("HEAD CANDIDATES");
        countCaption.setTextFill(Color.GRAY);
        countCaption.setFont(Font.font("System", FontWeight.BOLD, 10));
        countNumber.setFont(Font.font("System", FontWeight.BOLD, 36));
        countNumber.setTextFill(Color.web("#1d4ed8"));
        VBox badge = new VBox(0, countCaption, countNumber);
        badge.setAlignment(Pos.CENTER);
        badge.setPadding(new Insets(4, 22, 4, 22));
        badge.setStyle("-fx-background-color: #dbeafe; -fx-background-radius: 10;");

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox header = new HBox(16, titleBox, spacer, upload, badge);
        header.setAlignment(Pos.CENTER_LEFT);
        header.setPadding(new Insets(12, 20, 12, 20));
        header.setStyle("-fx-background-color: #f3f4f6;");

        // ---- controls ----
        ComboBox<HeadDetector.Method> method = new ComboBox<>();
        method.getItems().addAll(HeadDetector.Method.values());
        method.setValue(p.method);
        method.setMaxWidth(Double.MAX_VALUE);
        method.setOnAction(e -> { p.method = method.getValue(); refresh(); });

        ComboBox<HeadDetector.Threshold> thr = new ComboBox<>();
        thr.getItems().addAll(HeadDetector.Threshold.values());
        thr.setValue(p.threshold);
        thr.setMaxWidth(Double.MAX_VALUE);
        thr.setOnAction(e -> { p.threshold = thr.getValue(); refresh(); });

        stage.getItems().addAll("Annotated", "Grayscale", "Blurred", "Edges", "Binary", "Cleaned");
        stage.setValue("Annotated");
        stage.setMaxWidth(Double.MAX_VALUE);
        stage.setOnAction(e -> show());

        VBox general = group(true,
                "Method and view",
                field("Detection method", method),
                field("Threshold type", thr),
                field("View stage", stage));

        VBox pre = group(true,
                "Pre-processing",
                slider("Gaussian kernel", 1, 15, p.blur, v -> p.blur = v),
                slider("Global threshold", 0, 255, p.globalValue, v -> p.globalValue = v),
                slider("Adaptive block", 3, 101, p.adaptiveBlock, v -> p.adaptiveBlock = v),
                slider("Adaptive C", 0, 30, p.adaptiveC, v -> p.adaptiveC = v),
                slider("Canny low", 0, 255, p.cannyLow, v -> p.cannyLow = v),
                slider("Canny high", 0, 255, p.cannyHigh, v -> p.cannyHigh = v));

        VBox filters = group(true,
                "Region filters",
                slider("Min area", 10, 2000, (int) p.minArea, v -> p.minArea = v),
                slider("Max area", 100, 10000, (int) p.maxArea, v -> p.maxArea = v),
                slider("Min circularity %", 0, 100, (int) (p.minCircularity * 100), v -> p.minCircularity = v / 100.0),
                slider("Overlap (IoU) %", 5, 90, (int) (p.iou * 100), v -> p.iou = v / 100.0),
                slider("Min solidity %", 0, 100, (int) (p.minSolidity * 100), v -> p.minSolidity = v / 100.0),
                slider("Max eccentricity x10", 13, 100, (int) (p.maxEccentricity * 10), v -> p.maxEccentricity = v / 10.0));

        VBox blob = group(false,
                "Blob method",
                slider("Contrast threshold", 2, 100, p.blobThreshold, v -> p.blobThreshold = v),
                slider("Radius top row", 2, 40, p.blobRTop, v -> p.blobRTop = v),
                slider("Radius bottom row", 2, 80, p.blobRBottom, v -> p.blobRBottom = v),
                slider("Size tolerance %", 10, 150, (int) (p.blobTol * 100), v -> p.blobTol = v / 100.0));

        VBox hough = group(false,
                "Hough circles",
                slider("Accumulator", 8, 80, p.houghParam2, v -> p.houghParam2 = v),
                slider("Min radius", 2, 40, p.houghMinR, v -> p.houghMinR = v),
                slider("Max radius", 4, 80, p.houghMaxR, v -> p.houghMaxR = v),
                slider("Min distance", 3, 60, p.houghMinDist, v -> p.houghMinDist = v));

        VBox panel = new VBox(4, general, pre, filters, blob, hough);
        panel.setPadding(new Insets(6));
        ScrollPane left = new ScrollPane(panel);
        left.setFitToWidth(true);
        left.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        left.setPrefWidth(310);

        // ---- image area ----
        Label placeholder = new Label("Upload an image to begin");
        placeholder.setFont(Font.font("System", 16));
        placeholder.setTextFill(Color.web("#9ca3af"));
        view.setPreserveRatio(true);
        StackPane center = new StackPane(placeholder, view);
        center.setPadding(new Insets(10));
        center.setStyle("-fx-background-color: #1f2937;");
        view.fitWidthProperty().bind(center.widthProperty().subtract(20));
        view.fitHeightProperty().bind(center.heightProperty().subtract(20));
        placeholder.visibleProperty().bind(view.imageProperty().isNull());

        // ---- status bar ----
        status.setTextFill(Color.web("#374151"));
        status.setPadding(new Insets(7, 20, 7, 20));
        status.setMaxWidth(Double.MAX_VALUE);
        status.setStyle("-fx-background-color: #f3f4f6;");

        BorderPane root = new BorderPane(center);
        root.setTop(header);
        root.setLeft(left);
        root.setBottom(status);

        primary.setTitle("Classroom Student Counter");
        primary.setScene(new Scene(root, 1250, 800));
        primary.show();
    }

    /** A collapsible section that holds related controls. */
    private static VBox group(boolean expanded, String title, javafx.scene.Node... items) {
        VBox body = new VBox(8, items);
        body.setPadding(new Insets(8));
        TitledPane tp = new TitledPane(title, body);
        tp.setExpanded(expanded);
        tp.setAnimated(false);
        return new VBox(tp);
    }

    /** A grey caption above a control. */
    private static VBox field(String caption, javafx.scene.Node control) {
        Label l = new Label(caption);
        l.setTextFill(Color.GRAY);
        return new VBox(2, l, control);
    }

    /**
     * Creates a labelled slider that updates one parameter and re-runs detection when released.
     *
     * @param setter receives the new integer value
     */
    private VBox slider(String name, int min, int max, int value, Consumer<Integer> setter) {
        Label label = new Label(name + ": " + value);
        label.setFont(Font.font("System", 12));
        Slider s = new Slider(min, max, value);
        s.valueProperty().addListener((o, a, b) -> {
            int v = b.intValue();
            label.setText(name + ": " + v);
            setter.accept(v);
        });
        s.valueChangingProperty().addListener((o, was, now) -> { if (!now) refresh(); });
        s.setOnMouseReleased(e -> refresh());
        s.setOnKeyReleased(e -> refresh());
        return new VBox(2, label, s);
    }

    /**
     * Reads an image, downscales it to at most {@code MAX_WIDTH} pixels wide and runs detection.
     */
    private void load(File f) {
        Mat m = Imgcodecs.imread(f.getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
        if (m.empty()) { status.setText("Could not read " + f.getName()); return; }
        if (m.cols() > MAX_WIDTH) {
            double s = MAX_WIDTH / (double) m.cols();
            Imgproc.resize(m, m, new Size(), s, s, Imgproc.INTER_AREA);
        }
        image = m;
        refresh();
    }

    /**
     * Re-runs detection with the current parameters and updates the view and the count.
     */
    private void refresh() {
        if (image == null) return;
        try {
            result = detector.detect(image, p);
            show();
            countNumber.setText(String.valueOf(result.boxes().size()));
            status.setText("Head candidates: " + result.boxes().size()
                    + "   (contours found: " + result.rawContours() + ")");
        } catch (Exception ex) {
            status.setText("Detection failed: " + ex.getMessage());
        }
    }

    /**
     * Displays the stage chosen in the combo box.
     */
    private void show() {
        if (result == null) return;
        Mat m = switch (stage.getValue()) {
            case "Grayscale" -> result.gray();
            case "Blurred" -> result.blurred();
            case "Edges" -> result.edges();
            case "Binary" -> result.binary();
            case "Cleaned" -> result.cleaned();
            default -> HeadDetector.annotate(image, result.boxes());
        };
        MatOfByte buf = new MatOfByte();
        Imgcodecs.imencode(".png", m, buf);
        view.setImage(new Image(new ByteArrayInputStream(buf.toArray())));
    }

    public static void main(String[] args) { launch(args); }
}
