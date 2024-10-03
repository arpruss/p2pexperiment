package mobi.omegacentauri.p2pexperiment.calibration;

import android.hardware.Camera;
import android.util.Log;

import org.opencv.android.JavaCameraView;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

public class CameraCalibrator {
    private static final String TAG = "OCV::CameraCalibrator";

    private final Size mPatternSize = new Size(4, 11);
    private final int mCornersSize = (int)(mPatternSize.width * mPatternSize.height);
    private boolean mPatternWasFound = false;
    private MatOfPoint2f mCorners = new MatOfPoint2f();
    private List<Mat> mCornersBuffer = new ArrayList<>();
    private boolean mIsCalibrated = false;

    private Mat mCameraMatrix = new Mat();
    private Mat mDistortionCoefficients = new Mat();
    private double mRms;
    private double mSquareSize = 0.0181;
    private Size mImageSize;

    private boolean mAllowDistortion;

    public static double estimateFocalLength(JavaCameraView v, Size imageSize) {
        try {
            Field cameraField = JavaCameraView.class.getDeclaredField("mCamera");
            cameraField.setAccessible(true);
            Camera c = (Camera)cameraField.get(v);
            if (c == null)
                return 0;
            Camera.Parameters params = c.getParameters();
            if (params == null)
                return 0;
            double hva = params.getHorizontalViewAngle();
            if (hva == 0)
                return 0;
            return imageSize.width / 2. / (Math.tan(hva/2*Math.PI/180));
        }
        catch(NoSuchFieldException e) {
            return 0;
        } catch (IllegalAccessException e) {
            return 0;
        }
    }

    public static boolean estimateCameraMatrix(Mat cameraMatrix, JavaCameraView view, Size size, boolean tryHarder) {
        double fl = estimateFocalLength(view, size);
        if (fl == 0) {
            if (tryHarder)
                fl = size.width / 2 / (Math.tan(Math.PI/6));
            else
                return false;
        }

        cameraMatrix.put(0,0, new double [] {
                fl, 0, size.width/2,
                0, fl, size.height/2,
                0, 0, 1
        });

        return true;
    }

    public int getFlags() {
        int flags = Calib3d.CALIB_FIX_PRINCIPAL_POINT +
                Calib3d.CALIB_ZERO_TANGENT_DIST +
                Calib3d.CALIB_FIX_ASPECT_RATIO +
                Calib3d.CALIB_FIX_K4 +
                Calib3d.CALIB_FIX_K5;
        if (!mAllowDistortion)
            flags += Calib3d.CALIB_FIX_S1_S2_S3_S4+Calib3d.CALIB_FIX_K1+Calib3d.CALIB_FIX_K2+Calib3d.CALIB_FIX_K3+Calib3d.CALIB_FIX_S1_S2_S3_S4+Calib3d.CALIB_FIX_TAUX_TAUY;

        return flags;
    }

    public CameraCalibrator(int width, int height, boolean allowDistortion) {
        mAllowDistortion = allowDistortion;
        mImageSize = new Size(width, height);
        Mat.eye(3, 3, CvType.CV_64FC1).copyTo(mCameraMatrix);
        mCameraMatrix.put(0, 0, 1.0);
        //TODO: mCameraMatrix.put(1, 1, 1.0);
        Mat.zeros(5, 1, CvType.CV_64FC1).copyTo(mDistortionCoefficients);
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    public void processFrame(Mat grayFrame, Mat rgbaFrame) {
        findPattern(grayFrame);
        renderFrame(rgbaFrame);
    }

    public void calibrate(JavaCameraView view) {
        ArrayList<Mat> rvecs = new ArrayList<>();
        ArrayList<Mat> tvecs = new ArrayList<>();
        Mat reprojectionErrors = new Mat();
        ArrayList<Mat> objectPoints = new ArrayList<>();
        objectPoints.add(Mat.zeros(mCornersSize, 1, CvType.CV_32FC3));
        calcBoardCornerPositions(objectPoints.get(0));
        for (int i = 1; i < mCornersBuffer.size(); i++) {
            objectPoints.add(objectPoints.get(0));
        }

        int flags = getFlags();

        /*
        if (estimateCameraMatrix(mCameraMatrix, view, mImageSize, false)) {
            flags += Calib3d.CALIB_USE_INTRINSIC_GUESS;
            Log.v("Aruco", "using guess "+mCameraMatrix.dump());
        } */

        Calib3d.calibrateCamera(objectPoints, mCornersBuffer, mImageSize,
                mCameraMatrix, mDistortionCoefficients, rvecs, tvecs, flags);

        Log.v("Aruco", "computed "+mCameraMatrix.dump());

        mIsCalibrated = Core.checkRange(mCameraMatrix)
                && Core.checkRange(mDistortionCoefficients);

        mRms = computeReprojectionErrors(objectPoints, rvecs, tvecs, reprojectionErrors);
        Log.i(TAG, String.format("Average re-projection error: %f", mRms));
        Log.i(TAG, "Camera matrix: " + mCameraMatrix.dump());
        Log.i(TAG, "Distortion coefficients: " + mDistortionCoefficients.dump());


    }

    public void clearCorners() {
        mCornersBuffer.clear();
    }

    public void reset() {
        mIsCalibrated = false;
        CalibrationResult.setDefaults(mCameraMatrix,mDistortionCoefficients);
    }

    private void calcBoardCornerPositions(Mat corners) {
        final int cn = 3;
        float[] positions = new float[mCornersSize * cn];

        for (int i = 0; i < mPatternSize.height; i++) {
            for (int j = 0; j < mPatternSize.width * cn; j += cn) {
                positions[(int) (i * mPatternSize.width * cn + j + 0)] =
                        (2 * (j / cn) + i % 2) * (float) mSquareSize;
                positions[(int) (i * mPatternSize.width * cn + j + 1)] =
                        i * (float) mSquareSize;
                positions[(int) (i * mPatternSize.width * cn + j + 2)] = 0;
            }
        }
        corners.create(mCornersSize, 1, CvType.CV_32FC3);
        corners.put(0, 0, positions);
    }

    private double computeReprojectionErrors(List<Mat> objectPoints,
            List<Mat> rvecs, List<Mat> tvecs, Mat perViewErrors) {
        MatOfPoint2f cornersProjected = new MatOfPoint2f();
        double totalError = 0;
        double error;
        float[] viewErrors = new float[objectPoints.size()];

        MatOfDouble distortionCoefficients = new MatOfDouble(mDistortionCoefficients);
        int totalPoints = 0;
        for (int i = 0; i < objectPoints.size(); i++) {
            MatOfPoint3f points = new MatOfPoint3f(objectPoints.get(i));
            Calib3d.projectPoints(points, rvecs.get(i), tvecs.get(i),
                    mCameraMatrix, distortionCoefficients, cornersProjected);
            error = Core.norm(mCornersBuffer.get(i), cornersProjected, Core.NORM_L2);

            int n = objectPoints.get(i).rows();
            viewErrors[i] = (float) Math.sqrt(error * error / n);
            totalError  += error * error;
            totalPoints += n;
        }
        perViewErrors.create(objectPoints.size(), 1, CvType.CV_32FC1);
        perViewErrors.put(0, 0, viewErrors);

        return Math.sqrt(totalError / totalPoints);
    }

    private void findPattern(Mat grayFrame) {
        mPatternWasFound = Calib3d.findCirclesGrid(grayFrame, mPatternSize,
                mCorners, Calib3d.CALIB_CB_ASYMMETRIC_GRID);
    }

    public void addCorners() {
        if (mPatternWasFound) {
            mCornersBuffer.add(mCorners.clone());
        }
    }

    private void drawPoints(Mat rgbaFrame) {
        Calib3d.drawChessboardCorners(rgbaFrame, mPatternSize, mCorners, mPatternWasFound);
    }

    private void renderFrame(Mat rgbaFrame) {
        drawPoints(rgbaFrame);

        Imgproc.putText(rgbaFrame, "Captured: " + mCornersBuffer.size(), new Point(rgbaFrame.cols() / 3 * 2, rgbaFrame.rows() * 0.1),
                Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, new Scalar(255, 255, 0));
    }

    public Mat getCameraMatrix() {
        return mCameraMatrix;
    }

    public Mat getDistortionCoefficients() {
        return mDistortionCoefficients;
    }

    public int getCornersBufferSize() {
        return mCornersBuffer.size();
    }

    public double getAvgReprojectionError() {
        return mRms;
    }

    public boolean isCalibrated() {
        return mIsCalibrated;
    }

    public void setCalibrated() {
        mIsCalibrated = true;
    }

    public void setAllowDistortion(boolean allowDistortion) {
        mAllowDistortion = allowDistortion;
    }

    public boolean isAllowDistortion() {
        return mAllowDistortion;
    }
}
