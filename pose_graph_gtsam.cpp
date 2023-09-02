#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/slam/dataset.h>

using namespace std;
using namespace gtsam;

typedef struct 
{
    int id = 0;
    double t[3];
    double q[4]; // qx, qy, qz, qw
} State;

typedef struct
{
    int start_id, end_id;
    double t[3];
    double q[4]; // qx, qy, qz, qw

} Edge;

int main(int argc, char **argv) {

    string fin_path = "sphere.g2o";

    ifstream fin;
    fin.open(fin_path);
    assert(fin.is_open());

    NonlinearFactorGraph graph;
    Values initials;

    while(!fin.eof()){
        string name;
        fin >> name;
        if(name == "VERTEX_SE3:QUAT"){
            State state;
            fin >> state.id;
            for(int i = 0; i < 3; i++) fin >> state.t[i];
            for(int i = 0; i < 4; i++) fin >> state.q[i];
            initials.insert(Symbol('x', state.id), 
                            Pose3(Quaternion(state.q[3], state.q[0], state.q[1], state.q[2]), 
                                  Point3(state.t[0], state.t[1], state.t[2])));
        }

        else if(name == "EDGE_SE3:QUAT"){
            Edge edge;
            fin >> edge.start_id >> edge.end_id;
            /*  Somthing to add */
            for(int i = 0; i < 3; i++) fin >> edge.t[i];
            for(int i = 0; i < 4; i++) fin >> edge.q[i];
            Pose3 measurement = Pose3(Quaternion(edge.q[3], edge.q[0], edge.q[1], edge.q[2]),
                                      Point3(edge.t[0], edge.t[1], edge.t[2]));

            Matrix covariance(6, 6);
            for(int i = 0; i < 6 && fin.good(); i++){
                for(int j = i; j < 6 && fin.good(); j++){
                    fin >> covariance(i,j);
                    if(j != i) covariance(j,i) = covariance(i,j);
                }
            }

            noiseModel::Gaussian::shared_ptr gauss_model = noiseModel::Gaussian::Covariance(covariance);
            graph.add(BetweenFactor<Pose3>(Symbol('x', edge.start_id), Symbol('x', edge.end_id), measurement, gauss_model));
        }
    }
    fin.close();

    // fix first state
    auto prior_model = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    Key first_key = initials.keys()[0];
    Pose3 first_pose = initials.at<Pose3>(first_key);
    graph.addPrior(first_key, first_pose, prior_model);

    // initials.print("states are: ");
    // graph.print("edges:");

    GaussNewtonParams params;
    params.setMaxIterations(10);
    // params.setRelativeErrorTol(-1e+20);
    // params.setAbsoluteErrorTol(-1e+20);
    params.setVerbosity("ERROR");

    GaussNewtonOptimizer optimizer(graph, initials, params);
    Values results = optimizer.optimize();

    string filename = "results/gtsam_result.g2o";
    writeG2o(graph, results, filename);

    return 0;
}

