#include <iostream>
#include <eigen3/Eigen/Core>
#include <fstream>
#include <assert.h>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

using namespace std;
using namespace Eigen;
//using namespace ceres;

typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 6, 1> Vector6d;

typedef struct {
    int id = 0;
    double param[7] = {0};
    double se3[6] = {0};//(t,r)
} param_type;



Matrix6d JRInv(const Sophus::SE3d &e){
    Matrix6d J;
    J.block(0,0,3,3) = Sophus::SO3d::hat(e.so3().log());
    J.block(0,3,3,3) = Sophus::SO3d::hat(e.translation());
    J.block(3,0,3,3) = Matrix3d::Zero(3,3);
    J.block(3,3,3,3) = Sophus::SO3d::hat(e.so3().log());
    J = 0.5 * J + Matrix6d::Identity();
    return J;
//    return Matrix6d::Identity();
}


class SE3Parameterization : public ceres::LocalParameterization
{
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}

    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const
    {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

        Sophus::SE3d T = Sophus::SE3d::exp(lie);
        Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);

        // 李代数左乘更新
        Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

        for(int i = 0; i < 6; ++i)
            x_plus_delta[i] = x_plus_delta_lie(i, 0);

        return true;
    }
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const
    {
        ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
        return true;
    }
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};



class PoseGraphCostFunction : public ceres::SizedCostFunction<6,6,6>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    ~PoseGraphCostFunction(){}
    PoseGraphCostFunction(Sophus::SE3d _se3, Matrix6d _covariance): measurment_se3(_se3), convariance(_covariance){}

    virtual bool Evaluate(double const* const* parameters,
                          double *residuals,
                          double **jacobians) const{

        //Create SE3 with parameters
        Sophus::SE3d pose_i = Sophus::SE3d::exp(Vector6d(parameters[0]));
        Sophus::SE3d pose_j = Sophus::SE3d::exp(Vector6d(parameters[1]));
        Sophus::SE3d estimate = pose_i.inverse() * pose_j;

        //Get information matrix by LLT decomposition
        Matrix6d sqrt_info = Eigen::LLT<Matrix6d>(convariance.inverse()).matrixLLT().transpose();
        Eigen::Map<Vector6d> residual(residuals);
//        residual = sqrt_info * ((measurment_se3.inverse() * estimate).log());
        residual = (measurment_se3.inverse() * estimate).log();

        //Compute jacobians matrix
        if(jacobians){
            if(jacobians[0]) {
                Eigen::Map<Matrix6d> jacobian_i(jacobians[0]);
                Matrix6d J = JRInv(Sophus::SE3d::exp(residual));
                jacobian_i = (sqrt_info * (-J) * pose_j.inverse().Adj()).transpose();//雅各比矩阵的表达公式 J^T * info_matrix * J => J' = sqrt_info * J
            }

            if(jacobians[1]){
                Eigen::Map<Matrix6d> jacobian_j(jacobians[1]);
                Matrix6d J = JRInv(Sophus::SE3d::exp(residual));
                jacobian_j = (sqrt_info * J * pose_j.inverse().Adj()).transpose();
            }
        }
        residual = sqrt_info * ((measurment_se3.inverse() * estimate).log());//整体误差的表达公式 l(x) = e_ij^T * covariance * e_ij， 当delta_x -> 0的时候，l(x) -> 0
        return true;
    }
private:
    const Sophus::SE3d measurment_se3;
    const Matrix6d convariance;
};


void Convert2se3(param_type &_p){
    Quaterniond q(_p.param[6], _p.param[3], _p.param[4], _p.param[5]);
    Vector3d t(_p.param[0], _p.param[1], _p.param[2]);
    Vector6d se = Sophus::SE3d(q.normalized(),t).log();
//    auto tmp = Sophus::SE3(q,t).log();
//    auto tmp1 = Sophus::SE3::exp(tmp);
    for(int i = 0; i < 6; i++){
        _p.se3[i] = se(i, 0);
    }
}


int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);

    string fin_path = "/home/gao/Cpp_Projects/slambook2-master/ch10_ceres/sphere.g2o";

    ceres::Problem problem;
    vector<param_type> param;

    ifstream fin;
    fin.open(fin_path);
    assert(fin.is_open());
    ceres::LocalParameterization *local_param = new SE3Parameterization();
    while(!fin.eof()){
        string name;
        fin >> name;
        if(name == "VERTEX_SE3:QUAT"){
            param_type p;
            fin >> p.id;
            for(int i = 0; i < 7; i++) fin >> p.param[i];
            Convert2se3(p);
            param.push_back(p);

//            problem.AddParameterBlock(param.back().se3, 6, local_param);
        }
        else if(name == "EDGE_SE3:QUAT"){
            int vertex_i, vertex_j;
            fin >> vertex_i >> vertex_j;
            /*  Somthing to add */
            double m[7];//temporary measurement result
            for(int i = 0; i < 7; i++) fin >> m[i];
            Sophus::SE3d measurement(Quaternion<double>(m[6], m[3], m[4], m[5]).normalized(),
                                       Vector3d(m[0], m[1], m[2]));
            Matrix6d covariance;
            for(int i = 0; i < 6 && fin.good(); i++){
                for(int j = i; j < 6 && fin.good(); j++){
                    fin >> covariance(i,j);
                    if(j != i) covariance(j,i) = covariance(i,j);
                }
            }
            ceres::LossFunction *loss = new ceres::HuberLoss(1.0);
//            ceres::LossFunction *loss = nullptr;
            ceres::CostFunction *costfunc = new PoseGraphCostFunction(measurement, covariance);
            problem.AddResidualBlock(costfunc, loss, param[vertex_i].se3, param[vertex_j].se3);

            problem.SetParameterization(param[vertex_i].se3, local_param);
            problem.SetParameterization(param[vertex_j].se3, local_param);
        }
    }
    fin.close();

    cout << param.size() << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_linear_solver_iterations = 50;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;
//    std::cout << "Hello, World!" << std::endl;

//    std::cout<<summary.FullReport() <<std::endl;
    std::ofstream txt("/home/gao/Cpp_Projects/slambook2-master/ch10_ceres/result.g2o");
    for( int i=0; i < param.size(); i++ )
    {
        Eigen::Map<const Eigen::Matrix<double,6,1>> poseAVec6d( param[i].se3 );
        Sophus::SE3d poseSE3 = Sophus::SE3d::exp(poseAVec6d);
        Quaternion<double> q = poseSE3.so3().unit_quaternion();

        txt << "VERTEX_SE3:QUAT" << ' ';
        txt << i << ' ';
        txt << poseSE3.translation().transpose() << ' ';
        txt << q.x() <<' '<< q.y()<< ' ' << q.z() <<' '<< q.w()<<' ' << endl;
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


