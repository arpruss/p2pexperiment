package mobi.omegacentauri.p2pexperiment;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;
import java.lang.Math;
public class P2PExperiment {
    private final Mat m1_mat;
    private final Mat m2_mat;
    private final Mat m3_mat;
    private final Mat m4_mat;
    private final Boolean mVertical;
    //    private final Mat mCameraMatrix;
    private double focalLength;
    private Point cameraCenter;
    private final Mat marker1CameraVector;
    private final Mat marker2CameraVector;
    private static final double markerSize = 50;
    private static final double hSpacing = 150;
    private static final double vSpacing = 200;
    private double[] m1 = new double[] { 0, 0, 0 };
    private double[] m2 = new double[] { hSpacing, 0, 0 };
    private double[] m3 = new double[] { 0, vSpacing, 0 };
    private double[] m4 = new double[] { hSpacing, vSpacing, 0 };
    private final double d;
    private Mat vertical;
    private double beta;
    private int atMarker;
    private double rho1;
    private double rho2;
    private double h1,h2,d1,d2;
    private Mat cameraPosition;
    private Mat worldToCameraRotation;

    P2PExperiment(Mat cameraMatrix, Mat marker1, Mat marker2, double[] gravity, Boolean verticalMode) {
        mVertical = verticalMode;
        
        if (verticalMode) {
            m1 = new double[]{0, 0, 0};
            m2 = new double[]{hSpacing, 0, 0};
            m3 = new double[]{0, 0, vSpacing};
            m4 = new double[]{hSpacing, 0, vSpacing};
        }
        else {
            m1 = new double[]{0, 0, 0};
            m2 = new double[]{hSpacing, 0, 0};
            m3 = new double[]{0, vSpacing, 0};
            m4 = new double[]{hSpacing, vSpacing, 0};
        }

        d = Math.sqrt(Math.pow(m1[0]-m2[0],2)+Math.pow(m1[1]-m2[1],2));

        m1_mat = new Mat(3, 1,CvType.CV_64FC1);
        m1_mat.put(0,0,m1);
        Log.v("Aruco", "initializing 2");
        m2_mat = new Mat(3, 1,CvType.CV_64FC1);
        m2_mat.put(0,0,m2);
        m3_mat = new Mat(3, 1,CvType.CV_64FC1);
        m3_mat.put(0,0,m3);
        m4_mat = new Mat(3, 1,CvType.CV_64FC1);
        m4_mat.put(0,0,m4);

        focalLength = (cameraMatrix.get(0,0)[0]+cameraMatrix.get(1,1)[0])/2;
        cameraCenter = new Point(cameraMatrix.get(0,2)[0],cameraMatrix.get(1,2)[0]);
        marker1CameraVector = camera2DToVector(getQuadCenter(marker1));
        marker2CameraVector = camera2DToVector(getQuadCenter(marker2));
        
        Mat v0 = new Mat(3,1,CvType.CV_64FC1);
        v0.put(0,0,-gravity[1]);
        v0.put(1,0,-gravity[2]);
        v0.put(2,0,gravity[0]);
        vertical = new Mat(3,1,CvType.CV_64FC1);
        Core.normalize(v0, vertical);
        computeInputAngles();
        computeSolution();
        computePosition();
        computeRotation();
    }

    private Mat rotationAboutAxis(Mat axis, double angle) {
        double c = Math.cos(angle);
        double nc = 1-c;
        double s = Math.sin(angle);
        double x = axis.get(0,0)[0];
        double y = axis.get(1,0)[0];
        double z = axis.get(2,0)[0];
        Mat r = new Mat(3,3,CvType.CV_64FC1);
        r.put(0,0,new double []
                        {
                                x*x*nc+c, x*y*nc-z*s, x*z*nc+y*s,
                                x*y*nc+z*s, y*y*nc+c, y*z*nc-x*s,
                                x*z*nc-y*s, y*z*nc+x*s, z*z*nc+c
                        }
                );
        return r;
    }

    private Mat rotationVectorPairToVectorPair(Mat in1, Mat in2, Mat out1, Mat out2) {
        Log.v("Aruco", "pair to pair");
        Mat r11 = rotationVectorToVector(in1, out1);
        Mat in2r = r11.matMul(in2);
        double angle = angleAboutAxis(out1, in2r, out2);
        Mat r = rotationAboutAxis(out1, angle).matMul(r11);
        Log.v("Aruco", "test "+out1.dump()+" -> "+r.matMul(in1).dump());
        Log.v("Aruco", "test "+out2.dump()+" -> "+r.matMul(in2).dump());
        return r;
    }

    public static double angleAboutAxis(Mat normal, Mat v1, Mat v2) {
        Mat q1 = new Mat(3,1,CvType.CV_64FC1);
        double v1_dot_u = v1.dot(normal);
        Core.addWeighted(v1,1,normal,-v1_dot_u,0,q1);
        Mat q2 = new Mat(3,1,CvType.CV_64FC1);
        double v2_dot_u = v2.dot(normal);
        Core.addWeighted(v2,1,normal,-v2_dot_u,0,q2);
        return Math.atan2((q1.cross(q2)).dot(normal), q1.dot(q2));
    }

    private Mat rotationVectorToVector(Mat start, Mat end) {
        Mat normal = new Mat(3,1,CvType.CV_64FC1);
        Core.normalize(start.cross(end), normal);
        Log.v("Aruco", "start "+start.dump()+" end "+end.dump()+" normal "+normal.dump());
        double angle = angleAboutAxis(normal, start, end);
        Log.v("Aruco", "angle = "+angle);
        return rotationAboutAxis(normal, angle);
    }

    private void computeRotation() {
        Mat m = new Mat(3,0,CvType.CV_64FC1);
        Mat cameraToM1 = new Mat(3,0,CvType.CV_64FC1);
        Core.subtract(m1_mat,cameraPosition,m);
        Core.normalize(m,cameraToM1);
        Mat cameraToM2 = new Mat(3,0,CvType.CV_64FC1);
        Core.subtract(m2_mat,cameraPosition,m);
        Core.normalize(m,cameraToM2);
        worldToCameraRotation = rotationVectorPairToVectorPair(cameraToM1, cameraToM2, marker1CameraVector, marker2CameraVector);
    }

    private void computePosition() {
        double m2Angle = Math.atan2(m2[1]-m1[1],m2[0]-m1[0]);
        double xOffset = (d*d-d2*d2+d1*d1)/(2*d);
        double yOffset = Math.sqrt(4*d*d*d1*d1-Math.pow(d*d-d2*d2+d1*d1,2))/(2*d);
        if (beta<0)
            yOffset = -yOffset;
        double x = m1[0] + Math.cos(m2Angle)*xOffset - Math.sin(m2Angle)*yOffset;
        double y = m1[1] + Math.sin(m2Angle)*xOffset + Math.cos(m2Angle)*yOffset;
        double z = m1[2] + h1;
        cameraPosition = new Mat(3, 1, CvType.CV_64FC1);
        cameraPosition.put(0,0,x);
        cameraPosition.put(1,0,y);
        cameraPosition.put(2,0,z);
        Log.v("Aruco", "position "+cameraPosition.dump());
    }

    private void computeSolution() {
        boolean i_is_1 = rho1 != Math.PI/2;
        double rho_i, rho_j, delta;
        if (i_is_1) {
            rho_i = rho1;
            rho_j = rho2;
            delta = m2[2] - m1[2];
        }
        else {
            rho_i = rho2;
            rho_j = rho1;
            delta = m1[2] - m2[2];
        }
        double cot_j = 1./Math.tan(rho_j);
        double tan_i = Math.tan(rho_i);
        double cos_beta = Math.cos(beta);
        double cottan = cot_j * tan_i;
        double a = 1-2*cos_beta*cottan + cottan*cottan;
        double dj,di,hj,hi;
        if (m2[2] == m1[2]) {
            Log.v("Aruco", "rho_i "+rho_i+" rho_j "+rho_j);
            dj = d / Math.sqrt(a);
            di = dj * cot_j * tan_i;
            hj = -dj * cot_j;
            hi = hj;
        }
        else {
            double b = 2 * delta * tan_i * (cos_beta - cottan);
            double c = delta * delta * tan_i * tan_i - d * d;
            double sqrt_disc = Math.sqrt(b * b - 4 * a * c);
            dj = 0;
            di = 0;
            hi = 0;
            hj = 0;
            for (int s = -1 ; s <= 1 ; s += 2) {
                dj = (-b + s * sqrt_disc) / (2. * a);
                if (dj < 0) {
                    dj = 0;
                    continue;
                }
                hj =-dj * cot_j;
                hi = hj + delta;
                di = -(-dj * cot_j + delta) * tan_i;
            }
        }
        if (i_is_1) {
            d1 = di;
            d2 = dj;
            h1 = hi;
            h2 = hj;
        }
        else {
            d2 = di;
            d1 = dj;
            h2 = hi;
            h1 = hj;
        }
    }

    private Mat camera2DToVector(Point pt) {
        Mat v = new Mat(3,1, CvType.CV_64FC1);
        v.put(0,0,pt.x-cameraCenter.x);
        v.put(1,0,focalLength);
        v.put(2,0,-(pt.y-cameraCenter.y)); // OpenCV has upper right as origin
        Mat v1 = new Mat(3,1, CvType.CV_64FC1);
        Core.normalize(v,v1);
        return v1;
    }

    private Point cameraVectorTo2D(Mat v) {
        double t = focalLength / v.get(0,1)[0];
        return new Point(v.get(0,0)[0]*t+cameraCenter.x, -v.get(0,2)[0]*t+cameraCenter.y);
    }

    private static Point getQuadCenter(Mat quad) {
        // lines will be Ax+By=E (corner 0 to corner 2), Cx+Dy=F (corner 1 to corner 3),
        double A = quad.get(0,2)[1]-quad.get(0,0)[1]; // y-difference
        double B = -quad.get(0,2)[0]+quad.get(0,0)[0]; // x-difference
        double E = A*quad.get(0,0)[0]+B*quad.get(0,0)[1];
        double C = quad.get(0,3)[1]-quad.get(0,1)[1]; // y-difference
        double D = -quad.get(0,3)[0]+quad.get(0,1)[0]; // x-difference
        double F = C*quad.get(0,1)[0]+D*quad.get(0,1)[1];
        double det = A*D-B*C;
        return new Point( (-B * F + D * E)/det, (A * F - C * E)/det );
    }

    private void computeInputAngles() {
        Mat q1 = new Mat(3,1,CvType.CV_64FC1);
        // qi = vi (vi.u)u
        double v1_dot_u = marker1CameraVector.dot(vertical);
        Core.addWeighted(marker1CameraVector,1,vertical,-v1_dot_u,0,q1);
        Mat q2 = new Mat(3,1,CvType.CV_64FC1);
        double v2_dot_u = marker2CameraVector.dot(vertical);
        Core.addWeighted(marker2CameraVector,1,vertical,-v2_dot_u,0,q2);
        //\beta = \atantwo( (\mathbold q_1\times \mathbold q_2)\cdot u, \mathbold q_1\cdot\mathbold q_2 ).
        if (Core.hasNonZero(q1) && Core.hasNonZero(q2)) {
            beta = Math.atan2((q1.cross(q2)).dot(vertical), q1.dot(q2));
            atMarker = 0;
        }
        else {
            beta = 0;
            if (Core.hasNonZero(q1))
                atMarker = 2;
            else
                atMarker = 1;
        }
        rho1 = Math.acos(v1_dot_u);
        rho2 = Math.acos(v2_dot_u);
    }

    public List<List<Point>> getTestPositions() {
        List<List<Point>> out = new ArrayList<List<Point>>();
        out.add(markerWorldToCamera(m3));
        out.add(markerWorldToCamera(m4));
        return out;
    }

    private List<Point> markerWorldToCamera(double[] marker) {
        List<Point> out = new ArrayList<Point>();
        double dx = markerSize/2;
        double dy,dz;
        if (mVertical) {
            dy = 0;
            dz = markerSize/2;
        }
        else {
            dy = markerSize/2;
            dz = 0;
        }
        out.add(worldToCamera(marker[0]-dx, marker[1]-dy, marker[2]-dz));
        out.add(worldToCamera(marker[0]+dx, marker[1]-dy, marker[2]-dz));
        out.add(worldToCamera(marker[0]+dx, marker[1]+dy, marker[2]+dz));
        out.add(worldToCamera(marker[0]-dx, marker[1]+dy, marker[2]+dz));

        return out;
    }

    private Point worldToCamera(double x, double y, double z) {
        Mat v = new Mat(3,1,CvType.CV_64FC1);
        v.put(0,0,x);
        v.put(1,0,y);
        v.put(2,0,z);
        Mat vFromCamera = new Mat(3,1,CvType.CV_64FC1);
        Core.subtract(v, cameraPosition, vFromCamera);
        Mat vImage = worldToCameraRotation.matMul(vFromCamera);
        double x1 = cameraCenter.x + focalLength * vImage.get(0,0)[0] / vImage.get(1,0)[0];
        double y1 = cameraCenter.y - focalLength * vImage.get(2,0)[0] / vImage.get(1,0)[0];
        return new Point(x1,y1);
    }

}
