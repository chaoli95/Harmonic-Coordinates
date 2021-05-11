#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <float.h>
#include <igl/readOFF.h>
#include <igl//boundary_loop.h>
#include <igl/slice.h>
#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/unproject_on_plane.h>

// #define SIMPLEST_TRIANGULATION

using namespace std;

// original mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;
// cage mesh
Eigen::MatrixXd Vc;
Eigen::MatrixXi Fc;
Eigen::MatrixXi H;
// interior control mesh
Eigen::MatrixXd Vi;
Eigen::MatrixXi Fi;
// triangulates inside the cage
Eigen::MatrixXd Vt;
Eigen::MatrixXi Ft;
// weight function
Eigen::MatrixXd weight;
Eigen::MatrixXd harmonic_weight;
Eigen::MatrixXd mean_weight;
// used for display 
int current_cage_index = 0;
// used for deform
Eigen::Vector2d previous_mouse_coordinate;
int picked_cage_vertex;
bool doit = false;
double selection_threshold;
int original_mesh, cage_mesh;
unsigned int left_view, right_view;

void calculate_coordinate(igl::opengl::glfw::Viewer& viewer);
void update_mesh(igl::opengl::glfw::Viewer& viewer);

int nearest_control_vertex(Eigen::Vector3d &click_point)
{
  Eigen::RowVector2d click_point_2d(click_point(0), click_point(1));
  int cage_index;
  double cage_dist = (Vc.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&cage_index);
  int interior_index;
  double interior_dist = Vi.rows()>0 ? (Vi.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&interior_index) : DBL_MAX;
  double dist = std::min(cage_dist, interior_dist);
  if (dist > selection_threshold)
    return -1;
  int index = cage_dist < interior_dist ? cage_index : Vc.rows() + interior_index;
  return index;
}

bool callback_mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
  if (button == (int) igl::opengl::glfw::Viewer::MouseButton::Right)
  {
    // cout << "right click" << endl;
    return false;
  }
  bool click_left = viewer.current_mouse_x < viewer.core(left_view).viewport(3);
  Eigen::Vector3d Z;
  if (click_left)
  { 
    // example 708
    igl::unproject_on_plane(
      Eigen::Vector2i(viewer.current_mouse_x, viewer.core(left_view).viewport(3) - viewer.current_mouse_y),
      viewer.core(left_view).proj * viewer.core(left_view).view,
      viewer.core(left_view).viewport,
      Eigen::Vector4d(0,0,1,0),
      Z
    );
  } else 
  {
    igl::unproject_on_plane(
      Eigen::Vector2i(viewer.current_mouse_x, viewer.core(right_view).viewport(3) - viewer.current_mouse_y),
      viewer.core(right_view).proj * viewer.core(right_view).view,
      viewer.core(right_view).viewport,
      Eigen::Vector4d(0,0,1,0),
      Z
    ); 
  }
  int idx = nearest_control_vertex(Z);
  if (idx < 0)
    return false;
  current_cage_index = idx;
  if (click_left)
  {
    picked_cage_vertex = idx;
    previous_mouse_coordinate << Z(0), Z(1);
    doit = true;
  } else
  {
    viewer.data(cage_mesh).set_data(weight.row(current_cage_index));
    doit = false; 
    return true;
  }

  return doit;
}

bool callback_mouse_move(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y)
{
  if (!doit) return false;

  Eigen::Vector3d Z;
  igl::unproject_on_plane(
    Eigen::Vector2i(viewer.current_mouse_x, viewer.core(left_view).viewport(3) - viewer.current_mouse_y),
    viewer.core(left_view).proj * viewer.core(left_view).view,
    viewer.core(left_view).viewport,
    Eigen::Vector4d(0, 0, 1, 0),
    Z
  );
  Eigen::Vector2d current_mouse_coordinate(Z(0), Z(1));
  Eigen::Vector2d translation = current_mouse_coordinate - previous_mouse_coordinate;
  // cout << "before: " << Vc.row(picked_cage_vertex) << endl;
  previous_mouse_coordinate = current_mouse_coordinate;
  if (picked_cage_vertex < Vc.rows())
  {
    Vc.row(picked_cage_vertex) += translation;
  } else
  {
    Vi.row(picked_cage_vertex - Vc.rows()) += translation;
  }
  
  // cout << "after: " << Vc.row(picked_cage_vertex) << endl;
  calculate_coordinate(viewer);
  update_mesh(viewer);
  return true;
}

bool callback_mouse_up(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
  if (!doit)  return false;
  doit = false;
  picked_cage_vertex = -1;
  return true;
}

bool callback_key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifiers)
{
  bool handled = false;
  
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
    for (int j = 0; j < Vc.rows()+Vi.rows(); ++j)
    {
      if (j < Vc.rows())
        V.row(i) += harmonic_weight(j, i) * Vc.row(j);
      else 
        V.row(i) += harmonic_weight(j, i) * Vi.row(j-Vc.rows());
    }
  }
  // Vt << Vc, Vi, V;
}

void update_mesh(igl::opengl::glfw::Viewer& viewer) 
{
  viewer.data(original_mesh).clear();
  viewer.data(original_mesh).set_mesh(V, F);
  viewer.data(original_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(original_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  } 
  viewer.data(original_mesh).add_points(Vi, Eigen::RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(original_mesh).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      Eigen::RowVector3d(0,1,0)
    );
  }

  viewer.data(cage_mesh).set_data(weight.row(current_cage_index));
}

void calculate_harmonic_function() 
{
  // triangulate the cage
  int cage_vertex_num = Vc.rows()+Vi.rows();

  #ifdef SIMPLEST_TRIANGULATION
  Eigen::MatrixXd points(Vc.rows() + Vi.rows(), 2);
  points << Vc, Vi;
  igl::triangle::triangulate(points, Fc, H, "q", Vt, Ft);
  #else
  Eigen::MatrixXd points(Vc.rows() + Vi.rows() + V.rows(), 2);
  points << Vc, Vi, V;
  igl::triangle::triangulate(points, Fc, H, "", Vt, Ft);
  #endif

  cout << Vt.rows() << endl;
  cout << Vc.rows() << endl;
  cout << Vi.rows() << endl;
  cout << V.rows() << endl;

  // igl::triangle::triangulate(Vc, Fc, H, "q", Vt, Ft);

  weight.resize(Vc.rows()+Vi.rows(), Vt.rows());
  harmonic_weight.resize(Vc.rows()+Vi.rows(), V.rows());

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
  for (int i = 0; i < cage_vertex_num; ++i) 
  {
    phi.setZero();
    phi(i) = 1;
    // cout << Eigen::RowVectorXd(phi) << endl;
    // cout << -A_fc * phi << endl;
    Eigen::VectorXd h = solver.solve(-A_fc*phi);
    // weight.row(i) = h;
    for (int j = 0; j < Vt.rows(); ++j)
    {
      if (j < cage_vertex_num)
        weight(i, j) = phi(j);
      else
        weight(i, j) = h(j - cage_vertex_num);
    } 
    #ifndef SIMPLEST_TRIANGULATION
    harmonic_weight.row(i) << h.transpose();
    #endif
  }
  cout << weight.colwise().sum() << endl;
  #ifdef SIMPLEST_TRIANGULATION
  Eigen::MatrixXd P1, P2, P3;
  igl::slice(Vt, Ft.col(0), 1, P1);
  igl::slice(Vt, Ft.col(1), 1, P2);
  igl::slice(Vt, Ft.col(2), 1, P3);
  harmonic_weight.setZero();
  for (int idx = 0; idx < V.rows(); ++idx)
  {
    Eigen::MatrixXd P = V.row(idx).replicate(Ft.rows(), 1);
    Eigen::MatrixXd bcs;
    igl::barycentric_coordinates(P, P1, P2, P3, bcs);
    int triangle_idx = -1;
    for (int i = 0; i < Ft.rows(); ++i)
    {
      if (bcs(i,0)<=1 && bcs(i,0)>=0 &&
          bcs(i,1)<=1 && bcs(i,1)>=0 &&
          bcs(i,2)<=1 && bcs(i,2)>=0)
      {
        // cout << barycentric_coordinates.row(i) << endl;
        triangle_idx = i;
        break;
      }
    }
    assert(triangle_idx != -1);
    Eigen::RowVector3d bc = bcs.row(triangle_idx);
    // cout << bc << endl;
    for (int i = 0; i < weight.rows(); ++i)
    { 
      harmonic_weight(i, idx) = bc(0)*weight(i, Ft(triangle_idx, 0)) + bc(1)*weight(i, Ft(triangle_idx, 1)) + bc(2)*weight(i, Ft(triangle_idx, 2));
    }
  }
  #endif
  // cout << weight << endl;
}

void calculate_mean_coordinates_function()
{
  mean_weight.resize(Vc.rows(), V.rows());
  for (int i = 0; i < V.rows(); ++i)
  {
    Eigen::MatrixXd vectors = Vc.rowwise() - V.row(i);
    Eigen::MatrixX3d vectors3d;
    vectors3d.resize(vectors.rows(), 3);
    vectors3d.col(0) << vectors.col(0);
    vectors3d.col(1) << vectors.col(1);
    vectors3d.col(2).setZero();
    vectors3d.rowwise().normalize();
    // cout << vectors3d.rowwise().norm() << endl;
    Eigen::VectorXd tangents(vectors.rows());
    for (int j = 0; j < tangents.size(); ++j)
    {
      Eigen::Vector3d curr = vectors3d.row(j);
      Eigen::Vector3d next = vectors3d.row((j+1)%vectors3d.rows());
      Eigen::Vector3d middle = curr+next;
      double sin = curr.cross(middle).norm();
      double cos = curr.dot(middle);
      tangents(j) = sin/cos;
    }
    // cout << tangents << endl;
    for (int j = 0; j < Vc.rows(); ++j)
    {
      mean_weight(j,i) = (tangents(j) + tangents((j+Vc.rows()-1)%Vc.rows()))/vectors.row(j).norm();
    }
    mean_weight.col(i) /= mean_weight.col(i).sum();
  }
  // cout << mean_weight.colwise().sum() << endl;
}

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cout << "Usage harmonic-coordinates mesh.off cage.off" << std::endl;
    exit(0);
  }
  igl::readOFF(argv[1],V,F);
  assert(V.rows() > 0);
  // Eigen::MatrixXd Vc_tmp, Fc_tmp;
  igl::readOFF(argv[2], Vc, Fc);

  if (argc == 4)
  {
    igl::readOFF(argv[3], Vi, Fi);
    Vi.conservativeResize(Vi.rows(), 2);
    // cout << Fi << endl;
  } else
  {
    Vi.resize(0,2);
  }

  V.conservativeResize(V.rows(), 2);
  Vc.conservativeResize(Vc.rows(), 2);
  // Vc = Vc_tmp;

  // pre computation  
  calculate_harmonic_function();
  calculate_mean_coordinates_function();
  // calculate_coordinate();
  calculate_selection_threshold();
  // V.conservativeResize(V.rows(), 3);
  // V.col(2).setZero();

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  original_mesh = viewer.append_mesh(true);
  cage_mesh = viewer.append_mesh(true);
  viewer.data(original_mesh).set_mesh(V, F);
  viewer.data(cage_mesh).set_mesh(Vt, Ft);
  viewer.callback_init = [&](igl::opengl::glfw::Viewer &)
  {
    viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
    left_view = viewer.core_list[0].id;
    right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
    viewer.data(original_mesh).set_visible(false, right_view);
    viewer.data(cage_mesh).set_visible(false, left_view);
    return true;
  };
  viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) 
  {
    v.core(left_view).viewport = Eigen::Vector4f(0, 0, w / 2, h);
    v.core(right_view).viewport = Eigen::Vector4f(w / 2, 0, w - (w / 2), h);
    return true;
  };
  viewer.data(original_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(original_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  }
  viewer.data(original_mesh).add_points(Vi, Eigen::RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(original_mesh).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      Eigen::RowVector3d(0,1,0)
    );
  }

  viewer.data(cage_mesh).set_data(weight.row(current_cage_index));
  // viewer.data(cage_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  // for (int i = 0; i < Vc.rows(); ++i)
  // {
  //   viewer.data(cage_mesh).add_edges(
  //     Vc.row(Fc(i, 0)),
  //     Vc.row(Fc(i, 1)),
  //     Eigen::RowVector3d(1,0,0)
  //   );
  // }
  viewer.data(cage_mesh).add_points(Vi, Eigen::RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(cage_mesh).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      Eigen::RowVector3d(0,1,0)
    );
  }

  // viewer.data(cage_mesh).show_lines = false;
  
  // viewer.callback_key_down = callback_key_down;
  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;


  viewer.launch();
}
