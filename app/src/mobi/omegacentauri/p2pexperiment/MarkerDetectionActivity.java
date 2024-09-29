package mobi.omegacentauri.p2pexperiment;

import mobi.omegacentauri.p2pexperiment.calibration.CalibrationResult;

import org.opencv.android.CameraActivity;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import mobi.omegacentauri.p2pexperiment.calibration.CalibrationResult;
import mobi.omegacentauri.p2pexperiment.calibration.CameraCalibrationActivity;
import mobi.omegacentauri.p2pexperiment.calibration.CameraCalibrator;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import java.util.Collections;
import java.util.List;

public class MarkerDetectionActivity extends CameraActivity implements SensorEventListener,CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String  TAG = "QRdetection::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private MarkerProcessor mQRDetector;
    private MenuItem             mItemCalibration;
    private Mat mCameraMatrix;
    private Mat mDistortionCoefficients;

    double gravity[] = new double[3];
    private SensorManager sensorManager;
    private Sensor gravitySensor;
    private MenuItem mItemVertical;
    private SharedPreferences prefs;
    private Boolean mVertical = false;
    private static final String VERTICAL = "Vertical";


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        prefs = PreferenceManager.getDefaultSharedPreferences(this);
        mVertical = prefs.getBoolean(VERTICAL, false);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        gravitySensor = sensorManager.getDefaultSensor(Build.VERSION.SDK_INT >= 9 ? Sensor.TYPE_GRAVITY : Sensor.TYPE_ACCELEROMETER);

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            return;
        }

        Log.d(TAG, "Creating and setting view");
        mOpenCvCameraView = new JavaCameraView(this, -1);
        setContentView(mOpenCvCameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mQRDetector = new MarkerProcessor(true);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        sensorManager.unregisterListener(this);
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        gravity[0] = 0;
        gravity[1] = 0;
        gravity[2] = 0;
        sensorManager.registerListener(this, gravitySensor, SensorManager.SENSOR_DELAY_FASTEST);

        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.enableView();
        }
        mCameraMatrix = new Mat(CalibrationResult.CAMERA_MATRIX_ROWS+1, CalibrationResult.CAMERA_MATRIX_COLS, CvType.CV_64FC1);
        mDistortionCoefficients = new Mat(CalibrationResult.DISTORTION_COEFFICIENTS_SIZE, 1, CvType.CV_64FC1);
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemCalibration = menu.add("Calibration");
        mItemCalibration.setShowAsActionFlags(MenuItem.SHOW_AS_ACTION_ALWAYS);
        mItemVertical = menu.add("Vertical mode");
        mItemVertical.setShowAsActionFlags(MenuItem.SHOW_AS_ACTION_NEVER);
        mItemVertical.setCheckable(true);
        mItemVertical.setChecked(mVertical);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemCalibration) {
            Intent i = new Intent(this, CameraCalibrationActivity.class);
            startActivity(i);
        } else if (item == mItemVertical) {
            mVertical = ! mVertical;
            mItemCalibration.setChecked(mVertical);
            prefs.edit().putBoolean(VERTICAL, mVertical).apply();;
            return true;
        }

        return true;
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Log.v("Aruco", "camera frame");
        CalibrationResult.tryLoad(this, mCameraMatrix, mDistortionCoefficients, (JavaCameraView)mOpenCvCameraView, inputFrame.rgba().size());
        Mat renderedFrame = new Mat();
        if (Core.countNonZero(mDistortionCoefficients) > 0) {
            Calib3d.undistort(inputFrame.rgba(), renderedFrame,
                    mCameraMatrix, mDistortionCoefficients);
        }
        else {
            renderedFrame = inputFrame.rgba();
        }
        return mQRDetector.handleFrame(renderedFrame,mCameraMatrix,gravity,mVertical);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        gravity[0] = event.values[0];
        gravity[1] = event.values[1];
        gravity[2] = event.values[2];
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
