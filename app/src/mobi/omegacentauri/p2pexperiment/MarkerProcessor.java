package mobi.omegacentauri.p2pexperiment;

import org.opencv.core.Mat;
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
                                        Boolean verticalMode) {
        List<Mat> corners = new ArrayList<Mat>();
        Mat ids = new Mat();
        boolean result = findQRs(inputFrame, corners, ids);
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
                }
            }
        }
        for (Mat c : corners)
            c.release();
        ids.release();
        return inputFrame;
    }
}
