package mobi.omegacentauri.p2pexperiment.calibration;

import android.app.Activity;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.util.Log;

import org.opencv.android.JavaCameraView;
import org.opencv.core.Mat;
import org.opencv.core.Size;

public abstract class CalibrationResult {
    private static final String TAG = "OCV::CalibrationResult";

    public static final int CAMERA_MATRIX_ROWS = 3;
    public static final int CAMERA_MATRIX_COLS = 3;
    public static final int DISTORTION_COEFFICIENTS_SIZE = 5;

    public static void save(Activity activity, Mat cameraMatrix, Mat distortionCoefficients) {
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(activity);
        SharedPreferences.Editor editor = sharedPref.edit();

        if (cameraMatrix == null) {
            sharedPref.edit().remove("0").apply();
            return;
        }

        double[] cameraMatrixArray = new double[CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS];
        cameraMatrix.get(0,  0, cameraMatrixArray);
        for (int i = 0; i < CAMERA_MATRIX_ROWS; i++) {
            for (int j = 0; j < CAMERA_MATRIX_COLS; j++) {
                int id = i * CAMERA_MATRIX_ROWS + j;
                editor.putFloat(Integer.toString(id), (float)cameraMatrixArray[id]);
            }
        }

        double[] distortionCoefficientsArray = new double[DISTORTION_COEFFICIENTS_SIZE];
        distortionCoefficients.get(0, 0, distortionCoefficientsArray);
        int shift = CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS;
        for (int i = shift; i < DISTORTION_COEFFICIENTS_SIZE + shift; i++) {
            editor.putFloat(Integer.toString(i), (float)distortionCoefficientsArray[i-shift]);
        }

        editor.apply();
        Log.i(TAG, "Saved camera matrix: " + cameraMatrix.dump());
        Log.i(TAG, "Saved distortion coefficients: " + distortionCoefficients.dump());
    }

    public static void setDefaults(Mat cameraMatrix, Mat distortionCoefficients) {
        cameraMatrix.put(0,0,new double[] {1373.788111791253, 0, 959.5,
                0, 1373.788111791253, 539.5,
                0, 0, 1});
        Size size = distortionCoefficients.size();
        for (int i=0; i<size.height; i++)
            for (int j =0; j<size.width; j++)
                distortionCoefficients.put(i,j,0);
    }

    public static boolean tryLoad(Activity activity, Mat cameraMatrix, Mat distortionCoefficients) {
        return tryLoad(activity, cameraMatrix, distortionCoefficients, null, null);
    }

    public static boolean tryLoad(Activity activity, Mat cameraMatrix, Mat distortionCoefficients, JavaCameraView view, Size size) {
        setDefaults(cameraMatrix, distortionCoefficients);

        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(activity);
        if (sharedPref.getFloat("0", -1) == -1) {
            if (view != null) {
                CameraCalibrator.estimateCameraMatrix(cameraMatrix, view, size, true);
            }
//            Log.v("Aruco", "estimate "+cameraMatrix.dump());
            return false;
        }

        double[] cameraMatrixArray = new double[CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS];
        for (int i = 0; i < CAMERA_MATRIX_ROWS; i++) {
            for (int j = 0; j < CAMERA_MATRIX_COLS; j++) {
                int id = i * CAMERA_MATRIX_ROWS + j;
                cameraMatrixArray[id] = sharedPref.getFloat(Integer.toString(id), -1);
            }
        }
        cameraMatrix.put(0, 0, cameraMatrixArray);
//        Log.i("Aruco", "Loaded camera matrix: " + cameraMatrix.dump());

        if (! sharedPref.getBoolean(CameraCalibrationActivity.ALLOW_DISTORTION, true))
            return true;

        double[] distortionCoefficientsArray = new double[DISTORTION_COEFFICIENTS_SIZE];
        int shift = CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS;
        for (int i = shift; i < DISTORTION_COEFFICIENTS_SIZE + shift; i++) {
            distortionCoefficientsArray[i - shift] = sharedPref.getFloat(Integer.toString(i), -1);
        }
        distortionCoefficients.put(0, 0, distortionCoefficientsArray);
//        Log.i("Aruco", "Loaded distortion coefficients: " + distortionCoefficients.dump());

        return true;
    }

}
