#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <igl/readOFF.h>
#include <igl//boundary_loop.h>
#include <igl/slice.h>
#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/unproject_on_plane.h>

using namespace std;

// original mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;
// cage mesh
Eigen::MatrixXd Vc;
Eigen::MatrixXi Fc;
Eigen::MatrixXi H;
// triangulates inside the cage
Eigen::MatrixXd Vt;
Eigen::MatrixXi Ft;
// h_i(p)
Eigen::MatrixXd weights;
// used for display 
enum DisplayMode {ViewWeights, Deform};
DisplayMode display_mode = Deform;
int current_cage_index = 0;
// used for deform
Eigen::Vector2d previous_mouse_coordinate;
int picked_cage_vertex;
bool doit = false;
double selection_threshold;

void calculate_coordinate(igl::opengl::glfw::Viewer& viewer);

int nearest_cage_vertex(Eigen::Vector3d &click_point)
{
  Eigen::RowVector2d click_point_2d(click_point(0), click_point(1));
  int index;
  double dist = (Vc.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&index);
  if (dist > selection_threshold)
    index = -1;
  return index;
}

bool callback_mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
  if (button == (int) igl::opengl::glfw::Viewer::MouseButton::Right)
  {
    // cout << "right click" << endl;
    return false;
  }
  Eigen::Vector3d Z;
  // example 708
  igl::unproject_on_plane(
    Eigen::Vector2i(viewer.current_mouse_x, viewer.core().viewport(3) - viewer.current_mouse_y),
    viewer.core().proj * viewer.core().view,
    viewer.core().viewport,
    Eigen::Vector4d(0,0,1,0),
    Z
  );
  // cout << Z << endl;
  int idx = nearest_cage_vertex(Z);
  if (idx < 0)
    return false;
  if (display_mode == ViewWeights)
  {
    current_cage_index = idx;
    viewer.data().set_data(weights.row(current_cage_index));
    doit = false;
  }
  if (display_mode == Deform)
  {
    picked_cage_vertex = idx;
    previous_mouse_coordinate << Z(0), Z(1);
    doit = true;
  }

  return doit;
}

bool callback_mouse_move(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y)
{
  if (!doit) return false;

  Eigen::Vector3d Z;
  igl::unproject_on_plane(
    Eigen::Vector2i(viewer.current_mouse_x, viewer.core().viewport(3) - viewer.current_mouse_y),
    viewer.core().proj * viewer.core().view,
    viewer.core().viewport,
    Eigen::Vector4d(0, 0, 1, 0),
    Z
  );
  Eigen::Vector2d current_mouse_coordinate(Z(0), Z(1));
  Eigen::Vector2d translation = current_mouse_coordinate - previous_mouse_coordinate;
  // cout << "before: " << Vc.row(picked_cage_vertex) << endl;
  previous_mouse_coordinate = current_mouse_coordinate;
  Vc.row(picked_cage_vertex) += translation;
  
  // cout << "after: " << Vc.row(picked_cage_vertex) << endl;
  calculate_coordinate(viewer);
  return true;
}

bool callback_mouse_up(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
  if (!doit)  return false;
  doit = false;
  picked_cage_vertex = -1;
  return true;
}

// bool callback_pre_draw(igl::opengl::glfw::Viewer& viewer)
// {
//   if (display_mode == ViewWeights)
//   {

//   }
//   if (display_mode == Deform)
//   {

//   }
//   return false;
// }

bool callback_key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifiers)
{
  bool handled = false;
  if (key == '1' && display_mode != Deform)
  {
    display_mode = Deform;
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().add_points(Vc, Eigen::RowVector3d(1,0,0));
    for (int i = 0; i < Vc.rows(); ++i)
    {
      viewer.data().add_edges(
        Vc.row(Fc(i, 0)),
        Vc.row(Fc(i, 1)),
        Eigen::RowVector3d(1,0,0)
      );
    }
    viewer.data().set_face_based(true);
    handled = true;
  }
  if (key == '2' && display_mode != ViewWeights)
  {
    display_mode = ViewWeights;
    Vt << Vc, V;
    viewer.data().clear();
    viewer.data().set_mesh(Vt, Ft);
    viewer.data().set_data(weights.row(current_cage_index));
    viewer.data().set_face_based(false);
    handled = true;
  }
  // if (key == ' ')
  // {
  //   current_cage_index = (current_cage_index + 1) % Vc.rows();
  //   viewer.data().set_data(weights.row(current_cage_index));
  //   handled = true;
  // }
  
  return handled;
}

void calculate_selection_threshold()
{
  Eigen::MatrixXd starts, ends;
  igl::slice(Vc, Fc.col(0), 1, starts);
  igl::slice(Vc, Fc.col(1), 1, ends);
  double min_dist = (ends - starts).rowwise().norm().minCoeff();
  selection_threshold = min_dist / 5;
}

void calculate_coordinate(igl::opengl::glfw::Viewer& viewer) 
{
  V.setZero();
  for (int i = 0; i < V.rows(); ++i)
  {
    for (int j = 0; j < Vc.rows(); ++j)
    {
      V.row(i) += weights(j, i+Vc.rows()) * Vc.row(j);
    }
  }

  viewer.data().clear();
  viewer.data().set_mesh(V, F);
  viewer.data().add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data().add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  } 

  // double error = 0;
  // Eigen::MatrixXd tmp_V;
  // tmp_V.resizeLike(V);
  // for (int idx = 0; idx < V.rows(); ++idx)
  // {
  //   Eigen::MatrixXd P = V.row(idx).replicate(Ft.rows(), 1);
  //   Eigen::MatrixXd A, B, C;
  //   igl::slice(Vt, Ft.col(0), 1, A);
  //   igl::slice(Vt, Ft.col(1), 1, B);
  //   igl::slice(Vt, Ft.col(2), 1, C);
  //   Eigen::MatrixXd barycentric_coordinates;
  //   igl::barycentric_coordinates(P, A, B, C, barycentric_coordinates);
  //   int triangle_idx = -1;
  //   for (int i = 0; i < Ft.rows(); ++i)
  //   {
  //     if (barycentric_coordinates(i,0)<=1 && barycentric_coordinates(i,0)>=0 &&
  //         barycentric_coordinates(i,1)<=1 && barycentric_coordinates(i,1)>=0 &&
  //         barycentric_coordinates(i,2)<=1 && barycentric_coordinates(i,2)>=0)
  //     {
  //       // cout << barycentric_coordinates.row(i) << endl;
  //       triangle_idx = i;
  //       break;
  //     }
  //   }
  //   assert(triangle_idx != -1);
  //   Eigen::VectorXd new_v(2);
  //   new_v.setZero();
  //   for (int i = 0; i < weights.rows(); ++i)
  //   {
  //     Eigen::VectorXd D = weights.row(i).transpose();
  //     Eigen::RowVectorXd B = barycentric_coordinates.row(triangle_idx);
  //     Eigen::VectorXi I(1);
  //     I << triangle_idx;
  //     Eigen::MatrixXd tmp;
  //     igl::barycentric_interpolation(D, Ft, B, I, tmp);
  //     new_v += tmp(0,0) * Vc.row(i);
  //   }
  //   tmp_V.row(idx) = new_v;
  //   error += (V.row(idx) - Eigen::RowVectorXd(new_v)).norm();
  // }
  // cout << error << endl;
  // V = tmp_V;
}

void calculate_harmonic_function() 
{
  // triangulate the cage
  int cage_vertex_num = Vc.rows();

  Eigen::MatrixXd points(Vc.rows() + V.rows(), 2);
  points << Vc, V;
  igl::triangle::triangulate(points, Fc, H, "", Vt, Ft);

  // igl::triangle::triangulate(Vc, Fc, H, "q", Vt, Ft);

  weights.resize(Vc.rows(), Vt.rows());

  // Set up linear solver
  Eigen::VectorXi free_vertices, cage_vertices;
  cage_vertices.resize(cage_vertex_num);
  free_vertices.resize(Vt.rows() - cage_vertex_num);
  for (int i = 0; i < cage_vertex_num; ++i)
  {
    cage_vertices[i] = i;
  }
  for (int i = cage_vertex_num; i < Vt.rows(); ++i)
  {
    free_vertices[i-cage_vertex_num] = i;
  }

  // cout << Eigen::RowVectorXi(cage_vertices) << endl;
  // cout << Eigen::RowVectorXi(free_vertices) << endl;

  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(Vt, Ft, L);
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(Vt, Ft, igl::MASSMATRIX_TYPE_VORONOI, M);
  // cout << M << endl;
  Eigen::SparseMatrix<double> A;
  // A = (M.cwiseInverse()) * L;
  // L is negative semi-definite
  A = (-L).eval();
  
  Eigen::SparseMatrix<double> A_ff, A_fc;
  igl::slice(A, free_vertices, free_vertices, A_ff);
  igl::slice(A, free_vertices, cage_vertices, A_fc);
  // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_ff);
  assert(solver.info() == Eigen::Success);
  
  Eigen::VectorXd phi(cage_vertex_num);
  for (int i = 0; i < Vc.rows(); ++i) 
  {
    phi.setZero();
    phi(i) = 1;
    // cout << Eigen::RowVectorXd(phi) << endl;
    // cout << -A_fc * phi << endl;
    Eigen::VectorXd h = solver.solve(-A_fc*phi);
    for (int j = 0; j < Vt.rows(); ++j)
    {
      if (j < cage_vertex_num)
        weights(i, j) = phi(j);
      else
        weights(i, j) = h(j - cage_vertex_num);
    } 
    // Eigen::VectorXd tmp = weights.row(i);
    // cout << tmp.maxCoeff() << endl;
    // cout << tmp.minCoeff() << endl;
    // Eigen::RowVectorXd blah((M.cwiseInverse()) * A * tmp);
    // cout << blah.rows() << endl;
    // cout << blah.cols() << endl;
    // cout << blah.maxCoeff() << endl;
    // cout << blah.minCoeff() << endl;
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
  // Eigen::MatrixXd Vc_tmp, Fc_tmp;
  igl::readOFF(argv[2], Vc, Fc);
  cout << Fc << endl;

  V.conservativeResize(V.rows(), 2);
  Vc.conservativeResize(Vc.rows(), 2);
  // Vc = Vc_tmp;

  // pre computation  
  calculate_harmonic_function();
  // calculate_coordinate();
  calculate_selection_threshold();
  // V.conservativeResize(V.rows(), 3);
  // V.col(2).setZero();

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.callback_key_down = callback_key_down;
  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;

  viewer.data().set_mesh(V, F);
  viewer.data().add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data().add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  }
  viewer.data().set_face_based(true);

  viewer.launch();
}
