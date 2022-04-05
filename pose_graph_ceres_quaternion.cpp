//
// Created by gao on 2022/4/5.
//
#include <iostream>
#include <fstream>
#include <ceres/ceres.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <ceres/local_parameterization.h>

using namespace std;
using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;

typedef struct {
    int id = 0;
    double param[7] = {0};
    double t[3], q[4];
} param_type;


template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
{
    return q;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
{
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
    return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
{
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
    return ans;
}

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const{
        Eigen::Map<const Eigen::Quaterniond> _q(x);
        Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta));
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);
        q = (_q * dq).normalized();

        return true;
    }

    virtual bool ComputeJacobian(const double *x, double *jacobian) const{
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
        j.topRows<3>().setIdentity();
        j.bottomRows<1>().setZero();
        return true;
    }
    virtual int GlobalSize() const { return 4; };
    virtual int LocalSize() const { return 3; };
};

void Convert2qt(param_type &p){
    p.t[0] = p.param[0];
    p.t[1] = p.param[1];
    p.t[2] = p.param[2];

    p.q[0] = p.param[3]; // qx
    p.q[1] = p.param[4]; // qy
    p.q[2] = p.param[5]; // qz
    p.q[3] = p.param[6]; // qw
}


class PoseGraphCostFunction: public ceres::SizedCostFunction<6, 3, 4, 3, 4>{
public:
    PoseGraphCostFunction(Quaterniond q_m, Vector3d t_m, Matrix6d cov): q_m(q_m), t_m(t_m), covariance(cov){}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const{
        Vector3d t1(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quaterniond q1(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Vector3d t2(parameters[2][0], parameters[2][1], parameters[2][2]);
        Quaterniond q2(parameters[3][3], parameters[3][0], parameters[3][1], parameters[3][2]);

        Eigen::Map<Matrix<double, 6, 1>> residual(residuals);

        residual.segment(0, 3) = q1.inverse() * (t2 - t1) - t_m;
        residual.segment(3, 3) = 2 * (q_m.inverse() * (q1.inverse() * q2)).vec();

        if(jacobians){
            if(jacobians[0]){
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian(jacobians[0]);
                jacobian.block<3, 3>(0, 0) = -q1.inverse().toRotationMatrix();
                jacobian.block<3, 3>(3, 0) = Matrix3d::Zero();
            }

            if(jacobians[1]){
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jacobian(jacobians[1]);
                jacobian.setZero();
                jacobian.block<3, 3>(0, 0) = skewSymmetric(q1.inverse() * (t2 - t1));
                jacobian.block<3, 3>(3, 0) = -(Qright(q1.inverse() * q2) * Qleft(q_m.inverse())).bottomRightCorner<3, 3>();
//                jacobian.block<3, 3>(3, 0) = -0.5*(Qright(q2) * Qleft(q_m.inverse() * q1.inverse())).bottomRightCorner<3, 3>();
            }

            if(jacobians[2]){
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacobian(jacobians[2]);
                jacobian.block<3, 3>(0, 0) = q1.inverse().toRotationMatrix();
                jacobian.block<3, 3>(3, 0) = Matrix<double, 3, 3>::Zero();
            }

            if(jacobians[3]){
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jacobian(jacobians[3]);
                jacobian.setZero();
                jacobian.block<3, 3>(3, 0) = Qleft(q_m.inverse() * q1.inverse() * q2).bottomRightCorner<3, 3>();
//                jacobian.block<3, 3>(3, 0) = 0.5*(Qright(q2) * Qleft(q_m.inverse() * q1.inverse())).bottomRightCorner<3, 3>();
            }
        }
        return true;
    }

private:
    Quaterniond q_m;
    Vector3d t_m;
    Matrix6d covariance;
};


int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);

    string fin_path = "/home/gao/Cpp_Projects/slambook2-master/ch10_ceres/sphere.g2o";

    ceres::Problem problem;
    vector<param_type> param;

    ifstream fin;
    fin.open(fin_path);
//    ceres::LocalParameterization *local_param = new ceres::EigenQuaternionParameterization();
    ceres::LocalParameterization *local_param = new PoseLocalParameterization();

    while(!fin.eof()){
        string name;
        fin >> name;
        if(name == "VERTEX_SE3:QUAT"){
            param_type p;
            fin >> p.id;
            for(int i = 0; i < 7; i++) fin >> p.param[i];
            Convert2qt(p);
            param.push_back(p);
//            problem.AddParameterBlock(param.back().se3, 6, local_param);
        }
        else if(name == "EDGE_SE3:QUAT"){
            int vertex_i, vertex_j;
            fin >> vertex_i >> vertex_j;
            /*  Somthing to add */
            double m[7];//temporary measurement result
            for(int i = 0; i < 7; i++) fin >> m[i];
            Quaterniond q_m(m[6], m[3], m[4], m[5]);
            Vector3d t_m = {m[0], m[1], m[2]};

            Matrix6d covariance;
            for(int i = 0; i < 6 && fin.good(); i++){
                for(int j = i; j < 6 && fin.good(); j++){
                    fin >> covariance(i,j);
                    if(j != i) covariance(j,i) = covariance(i,j);
                }
            }
            ceres::LossFunction *loss = new ceres::HuberLoss(1.0);
//            ceres::LossFunction *loss = nullptr;
            ceres::CostFunction *costfunc = new PoseGraphCostFunction(q_m, t_m, covariance);
            problem.AddResidualBlock(costfunc, loss,
                                     param[vertex_i].t, param[vertex_i].q,
                                     param[vertex_j].t, param[vertex_j].q);

            problem.SetParameterization(param[vertex_i].q, local_param);
            problem.SetParameterization(param[vertex_j].q, local_param);
        }
    }
    fin.close();

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, & summary);

    cout << summary.BriefReport() << endl;


    std::ofstream txt("/home/gao/Cpp_Projects/slambook2-master/ch10_ceres/qua_result.g2o");
    for( int i=0; i < param.size(); i++ )
    {
        txt << "VERTEX_SE3:QUAT" << ' ';
        txt << i << ' ';
        txt << param[i].t[0] << ' ' << param[i].t[1] << ' ' << param[i].t[2] << ' ' ;
        txt << param[i].q[0] << ' ' << param[i].q[1] << ' ' << param[i].q[2] << ' ' << param[i].q[3] << endl;
    }
    fin.open(fin_path);
    while(!fin.eof()){
        string s;
        getline(fin, s);
        if(s[0] != 'E') continue;
        else txt << s << endl;
    }
    fin.close();
    txt.close();
    return 0;

}
