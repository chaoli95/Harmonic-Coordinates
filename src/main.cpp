#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <igl/readOFF.h>
#include <igl//boundary_loop.h>
#include <igl/slice.h>
#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

using namespace std;

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd Vc;
Eigen::MatrixXi Fc;
Eigen::MatrixXi H;
Eigen::MatrixXd V2;
Eigen::MatrixXi F2;
Eigen::MatrixXd weights;
int current_cage_index = 0;

bool callback_key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifiers)
{
  bool handled = false;
  if (key == ' ')
  {
    current_cage_index = (current_cage_index + 1) % Vc.rows();
    viewer.data().set_data(weights.row(current_cage_index));
    handled = true;
  }
  
  return handled;
}

void calculate_harmonic_coordinates() 
{
  // triangulate the cage
  int cage_vertex_num = Vc.rows();
  Fc.resize(Vc.rows(), 2);
  for (int i=0; i<Vc.rows(); ++i)
  {
    Fc(i, 0) = i;
    Fc(i, 1) = (i+1)%Vc.rows();
  }
  igl::triangle::triangulate(Vc, Fc, H, "a40q", V2, F2);

  weights.resize(Vc.rows(), V2.rows());

  Eigen::VectorXi free_vertices, cage_vertices;
  cage_vertices.resize(cage_vertex_num);
  free_vertices.resize(V2.rows() - cage_vertex_num);
  for (int i = 0; i < V2.rows(); ++i)
  {
    if (i < cage_vertex_num)
      cage_vertices[i] = i;
    else
      free_vertices[i - cage_vertex_num] = i;
  }

  // Set up Linear Solver
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V2, F2, L);
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V2, F2, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  Eigen::SparseMatrix<double> A;
  A = (M.cwiseInverse()) * L;
  // A = L;
  
  Eigen::SparseMatrix<double> A_ff, A_fc;
  igl::slice(A, free_vertices, free_vertices, A_ff);
  igl::slice(A, free_vertices, cage_vertices, A_fc);
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_ff);
  assert(solver.info() == Eigen::Success);
  Eigen::VectorXd phi(cage_vertex_num);
  for (int i = 0; i < Vc.rows(); ++i) 
  {
    phi.setZero();
    phi(i) = 1;
    cout << Eigen::RowVectorXd(phi) << endl;
    Eigen::VectorXd h = solver.solve(-A_fc*phi);
    // Eigen::VectorXd C;
    // C.setZero(V2.rows());
    // cout << Eigen::RowVectorXd(h) << endl;
    // cout << h.minCoeff() << h.maxCoeff() << endl;
    int free_index = 0;
    int cage_index = 0;
    for (int j = 0; j < V2.rows(); ++j)
    {
      if (cage_index < cage_vertices.size() && j == cage_vertices[cage_index])
        weights(i, j) = phi(cage_index++);
      else
        weights(i, j) = h(free_index++);
    } 
    cout << weights(i, 0) << endl;
    Eigen::VectorXd tmp = weights.row(i);
    cout << tmp.maxCoeff() << endl;
    cout << tmp.minCoeff() << endl;
    Eigen::RowVectorXd blah((M.cwiseInverse()) * A * tmp);
    cout << blah.maxCoeff() << endl;
    cout << blah.minCoeff() << endl;
    // cout << Eigen::RowVectorXd(weights.row(i)) << endl;
  }
  
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cout << "Usage harmonic-coordinates mesh.off cage.off" << std::endl;
    exit(0);
  }
  igl::readOFF(argv[1],V,F);
  assert(V.rows() > 0);

  igl::readOFF(argv[2], Vc, Fc);

  V.conservativeResize(V.rows(), 2);
  Vc.conservativeResize(Vc.rows(), 2);
  
  calculate_harmonic_coordinates();

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.callback_key_down = callback_key_down;
  viewer.data().set_mesh(V2, F2);
  viewer.data().set_data(weights.row(current_cage_index));
  // cout << Eigen::RowVectorXd(weights.row(6)) << endl;
  viewer.data().set_face_based(true);

  viewer.launch();
}
