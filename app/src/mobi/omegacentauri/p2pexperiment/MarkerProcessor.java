package mobi.omegacentauri.p2pexperiment;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.DetectorParameters;

import java.util.ArrayList;
import java.util.List;

public class MarkerProcessor {
    private ArucoDetector detector;
    private static final String TAG = "QRProcessor";
    private Scalar SourceColor = new Scalar(0, 255, 0);
    private Scalar DestColor = new Scalar(255, 0, 0);
    private Scalar FontColor = new Scalar(0, 0, 255);

    public MarkerProcessor(boolean useArucoDetector) {
        detector = new ArucoDetector();
        DetectorParameters par = detector.getDetectorParameters();
        par.set_maxErroneousBitsInBorderRate(.1);
        par.set_useAruco3Detection(true);
        detector.setDetectorParameters(par);
    }

    private boolean findQRs(Mat inputFrame, List<Mat> corners, Mat ids) {
        detector.detectMarkers(inputFrame, corners, ids);
        return true;
    }

    private void renderQRs(Mat inputFrame, List<Mat> corners, Mat ids) {
        int i = 0;
        for (Mat points : corners) {
            for (int j = 0; j < points.cols(); j++) {
                Point pt1 = new Point(points.get(0, j));
                Point pt2 = new Point(points.get(0, (j + 1) % 4));
                Imgproc.line(inputFrame, pt1, pt2, SourceColor, 3);
            }
            /*double[] pos = points.get(0,0);
            Point pt = new Point(pos[0],pos[1]-5); //points.get(0,0));
            String id = Integer.toString((int) ids.get(i,0)[0]);
            Imgproc.putText(inputFrame,id,pt,
                    0,1,SourceColor); */
            i++;
        }
    }

    /* this method to be called from the outside. It processes the frame to find QR codes. */
    public synchronized Mat handleFrame(Mat inputFrame, Mat cameraMatrix, double[] gravity,
                                        Boolean verticalMode, Boolean myColorFilter) {
        List<Mat> corners = new ArrayList<Mat>();
        Mat ids = new Mat();

        Mat filtered;
        if (myColorFilter) {
            Core.multiply(inputFrame, new Scalar(0, 1, 1), inputFrame);
            double blackPoint = .4;
            double whitePoint = .7;
            Core.addWeighted(inputFrame, 1 / (whitePoint - blackPoint), inputFrame, 0, -blackPoint * 255 / (whitePoint - blackPoint), inputFrame);
            filtered = new Mat();
            Imgproc.cvtColor(inputFrame, filtered, Imgproc.COLOR_RGBA2GRAY, 1);
        }
        else {
            filtered = inputFrame;
        }
        boolean result = findQRs(filtered, corners, ids);
        if (result) {
            renderQRs(inputFrame, corners, ids);

            if (cameraMatrix != null && ( gravity[0] != 0 || gravity[1] != 0 || gravity[2] != 0)) {
                Mat marker1 = null;
                Mat marker2 = null;
                Mat marker3 = null;
                Mat marker4 = null;
                for (int i = 0; i < corners.size(); i++) {
                    if (ids.get(i, 0)[0] == 1)
                        marker1 = corners.get(i);
                    else if (ids.get(i, 0)[0] == 2)
                        marker2 = corners.get(i);
                    else if (ids.get(i, 0)[0] == 3)
                        marker3 = corners.get(i);
                    else if (ids.get(i, 0)[0] == 4)
                        marker4 = corners.get(i);
                }
                if (marker1 != null && marker2 != null) {
                    P2PExperiment experiment = new P2PExperiment(cameraMatrix, marker1, marker2, marker3, marker4, gravity, verticalMode);
                    List<List<Point>> out = experiment.getTestPositions();
                    for (List<Point> p : out)
                        for (int i = 0 ; i < 4 ; i++) {
                            Imgproc.line(inputFrame, p.get(i), p.get((i + 1) % 4), DestColor, 3);
                        }
                    int rotationIndex = getRotation(gravity);
                    if (rotationIndex != -1)
                        Core.rotate(inputFrame, inputFrame, rotationIndex);
                    Mat darken = new Mat(inputFrame, new Rect(0,0,900,150));
                    Core.multiply(darken, new Scalar(0.5,0.5,0.5), darken);
                    String s = String.format("P2PA: %.1f,%.1f,%.1f",
                            experiment.cameraPosition.get(0,0)[0],
                            experiment.cameraPosition.get(1,0)[0],
                            experiment.cameraPosition.get(2,0)[0]
                            );
                    Imgproc.putText(inputFrame,s,new Point(5,60),
                            0,2,SourceColor);
                    if (experiment.cameraPositionP16P != null) {
                        s = String.format("P16P: %.1f,%.1f,%.1f",
                                experiment.cameraPositionP16P.get(0, 0)[0],
                                experiment.cameraPositionP16P.get(1, 0)[0],
                                experiment.cameraPositionP16P.get(2, 0)[0]
                        );
                        Imgproc.putText(inputFrame, s, new Point(5, 120),
                                0, 2, SourceColor);
                    }
                    if (rotationIndex != -1) {
                        Core.rotate(inputFrame, inputFrame, 2-rotationIndex);
                    }
                }
            }
        }
        for (Mat c : corners)
            c.release();
        ids.release();
        return inputFrame;
    }

    private static int getRotation(double[] gravity) {
        double ax = Math.abs(gravity[0]);
        double ay = Math.abs(gravity[1]);
        double az = Math.abs(gravity[2]);
        if (az >= ax && az >= ay) {
            return -1;
        }
        if (ax >= ay) {
            if (gravity[0]>=0) {
                return -1;
            }
            else {
                return Core.ROTATE_180;
            }
        }
        else {
            if (gravity[1]>=0) {
                return Core.ROTATE_90_CLOCKWISE;
            }
            else {
                return Core.ROTATE_90_COUNTERCLOCKWISE;
            }
        }
    }
}
