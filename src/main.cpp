#include <iostream>
#include <float.h>
#include <math.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/unproject_on_plane.h>

// #define SIMPLE_TRIANGULATION

using namespace std;

// original mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;
// cage mesh
Eigen::MatrixXd original_Vc;
Eigen::MatrixXd Vc;
Eigen::MatrixXi Fc;
Eigen::MatrixXi H;
// interior control mesh
Eigen::MatrixXd original_Vi;
Eigen::MatrixXd Vi;
Eigen::MatrixXi Fi;
// triangulates inside the cage
Eigen::MatrixXd Vt;
Eigen::MatrixXi Ft;
// weight function
Eigen::MatrixXd h_weight_in_cage;
Eigen::MatrixXd h_weight;
Eigen::MatrixXd m_weight_in_cage;
Eigen::MatrixXd m_weight;
// used for display 
int current_cage_index = 0;
// used for deform
Eigen::Vector2d previous_mouse_coordinate;
int picked_cage_vertex;
bool doit = false;
double selection_threshold;
int original_mesh, h_weight_mesh, m_weight_mesh;
unsigned int left_view, right_view;
enum CoordinateType {Harmonic, Mean};
CoordinateType coordinate_type = Harmonic;

void calculate_coordinate();
void update_mesh(igl::opengl::glfw::Viewer& viewer);
void set_original_mesh(igl::opengl::glfw::Viewer& viewer);

int nearest_control_vertex(Eigen::Vector3d &click_point, bool original_cage)
{
  Eigen::RowVector2d click_point_2d(click_point(0), click_point(1));
  Eigen::MatrixXd& cage = original_cage ? original_Vc : Vc;
  Eigen::MatrixXd& interor = original_cage? original_Vi : Vi;
  int cage_index;
  double cage_dist = (cage.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&cage_index);
  int interior_index;
  double interior_dist = interor.rows()>0&&coordinate_type==Harmonic ? (interor.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&interior_index) : DBL_MAX;
  double dist = std::min(cage_dist, interior_dist);
  if (dist > selection_threshold)
    return -1;
  int index = cage_dist < interior_dist ? cage_index : cage.rows() + interior_index;
  return index;
}

bool callback_mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
  if (button == (int) igl::opengl::glfw::Viewer::MouseButton::Right)
  {
    // cout << "right click" << endl;
    return false;
  }
  bool click_left = viewer.current_mouse_x < viewer.core(left_view).viewport(2);
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
  int idx = nearest_control_vertex(Z, !click_left);
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
    if (coordinate_type==Harmonic)
      viewer.data(h_weight_mesh).set_data(h_weight_in_cage.row(current_cage_index));
    else
      viewer.data(m_weight_mesh).set_data(m_weight_in_cage.row(current_cage_index));
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
  calculate_coordinate();
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
  if (key == '1' && coordinate_type != Harmonic)
  {
    coordinate_type = Harmonic;
    calculate_coordinate();
    viewer.data(m_weight_mesh).set_visible(false, right_view);
    viewer.data(h_weight_mesh).set_visible(true, right_view);
    update_mesh(viewer);
  } 
  if (key == '2' && coordinate_type != Mean && Vi.rows() == 0)
  {
    coordinate_type = Mean;
    calculate_coordinate();
    viewer.data(h_weight_mesh).set_visible(false, right_view);
    viewer.data(m_weight_mesh).set_visible(true, right_view);;
    update_mesh(viewer);
  }
  return handled;
}

// should have a better way
void calculate_selection_threshold()
{
  Eigen::MatrixXd starts, ends;
  igl::slice(Vc, Fc.col(0), 1, starts);
  igl::slice(Vc, Fc.col(1), 1, ends);
  double min_dist = (ends - starts).rowwise().norm().minCoeff();
  selection_threshold = min_dist / 3;
}

void calculate_coordinate() 
{
  V.setZero();
  for (int i = 0; i < V.rows(); ++i)
  {
    if (coordinate_type == Harmonic)
    {
      for (int j = 0; j < Vc.rows()+Vi.rows(); ++j)
      {
        if (j < Vc.rows())
          V.row(i) += h_weight(j, i) * Vc.row(j);
        else 
          V.row(i) += h_weight(j, i) * Vi.row(j-Vc.rows());
      }
    } else 
    {
      for (int j = 0; j < Vc.rows(); ++j)
      {
        V.row(i) += m_weight(j, i) * Vc.row(j);
      }
    }
  }
}

void set_original_mesh(igl::opengl::glfw::Viewer& viewer)
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
}

void update_mesh(igl::opengl::glfw::Viewer& viewer) 
{
  set_original_mesh(viewer);
  if (coordinate_type == Harmonic) 
    viewer.data(h_weight_mesh).set_data(h_weight_in_cage.row(current_cage_index));
  else 
    viewer.data(m_weight_mesh).set_data(m_weight_in_cage.row(current_cage_index));
}

void calculate_harmonic_function() 
{
  // triangulate the cage
  int control_vertex_num = Vc.rows()+Vi.rows();

  #ifdef SIMPLE_TRIANGULATION
  Eigen::MatrixXd points(Vc.rows() + Vi.rows(), 2);
  points << Vc, Vi;
  igl::triangle::triangulate(points, Fc, H, "q", Vt, Ft);
  #else
  Eigen::MatrixXd points(Vc.rows() + Vi.rows() + V.rows(), 2);
  points << Vc, Vi, V;
  igl::triangle::triangulate(points, Fc, H, "", Vt, Ft);
  #endif

  h_weight_in_cage.resize(Vc.rows()+Vi.rows(), Vt.rows());
  h_weight.resize(Vc.rows()+Vi.rows(), V.rows());

  // Set up linear solver
  Eigen::VectorXi free_vertices, cage_vertices;
  cage_vertices = Eigen::VectorXi::LinSpaced(control_vertex_num, 0, control_vertex_num-1);
  free_vertices = Eigen::VectorXi::LinSpaced(Vt.rows()-control_vertex_num, 
                                              control_vertex_num, Vt.rows()-1);
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(Vt, Ft, L);

  // Eigen::SparseMatrix<double> M;
  // igl::massmatrix(Vt, Ft, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  Eigen::SparseMatrix<double> A;
  // L is negative semi-definite
  A = (-L).eval();
  // A = ((M.cwiseInverse()) * (-L)).eval();

  Eigen::SparseMatrix<double> A_ff, A_fc;
  igl::slice(A, free_vertices, free_vertices, A_ff);
  igl::slice(A, free_vertices, cage_vertices, A_fc);
  // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_ff);
  assert(solver.info() == Eigen::Success);
  Eigen::VectorXd phi(control_vertex_num);
  for (int i = 0; i < control_vertex_num; ++i) 
  {
    phi.setZero();
    phi(i) = 1;
    Eigen::VectorXd h = solver.solve(-A_fc*phi);
    // weight.row(i) = h;
    for (int j = 0; j < Vt.rows(); ++j)
    {
      if (j < control_vertex_num)
        h_weight_in_cage(i, j) = phi(j);
      else
        h_weight_in_cage(i, j) = h(j - control_vertex_num);
    } 
    #ifndef SIMPLE_TRIANGULATION
    // if (i < h_weight.rows())
    //   for (int j = 0; j < h_weight.cols(); ++j)
    //     h_weight(i,j) = h(j);
      // h_weight.row(i) << h.transpose();
    #endif
  }
  // #ifdef SIMPLE_TRIANGULATION
  Eigen::MatrixXd P1, P2, P3;
  igl::slice(Vt, Ft.col(0), 1, P1);
  igl::slice(Vt, Ft.col(1), 1, P2);
  igl::slice(Vt, Ft.col(2), 1, P3);
  h_weight.setZero();
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
    for (int i = 0; i < h_weight_in_cage.rows(); ++i)
    { 
      h_weight(i, idx) = bc(0)*h_weight_in_cage(i, Ft(triangle_idx, 0)) + bc(1)*h_weight_in_cage(i, Ft(triangle_idx, 1)) + bc(2)*h_weight_in_cage(i, Ft(triangle_idx, 2));
    }
  }
  // #endif
  // cout << h_weight_in_cage.colwise().sum() << endl;
}

void calculate_mean_coordinates_function(Eigen::MatrixXd &V, Eigen::MatrixXd &cage, Eigen::MatrixXd &weight)
{
  weight.resize(cage.rows(), V.rows());
  for (int i = 0; i < V.rows(); ++i)
  {
    Eigen::MatrixXd vectors = cage.rowwise() - V.row(i);
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
      bool positive = curr(0)*next(1)-curr(1)*next(0)<0;
      // cout << positive << endl;
      Eigen::Vector3d middle = (curr+next);
      double sin = curr.cross(middle).norm();
      double cos = curr.dot(middle);
      tangents(j) = positive ? sin/cos : -sin/cos;
    }
    // cout << tangents << endl;
    for (int j = 0; j < cage.rows(); ++j)
    {
      weight(j,i) = (tangents(j) + tangents((j+cage.rows()-1)%cage.rows()))/vectors.row(j).norm();
      // if (weight(j,i) < 0)
      // {
      //   cout << "negative weight" << endl;
      // }
    }
    weight.col(i) /= weight.col(i).sum();
  }
  // cout << weight.colwise().sum() << endl;
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

  V.conservativeResize(V.rows(), 2);
  Vc.conservativeResize(Vc.rows(), 2);
  original_Vc.resizeLike(Vc);
  original_Vc << Vc;

  if (argc == 4)
  {
    igl::readOFF(argv[3], Vi, Fi);
    Vi.conservativeResize(Vi.rows(), 2);
    // cout << Fi << endl;
  } else
  {
    Vi.resize(0,2);
  }
  original_Vi.resizeLike(Vi);
  original_Vi << Vi;

  // pre computation  
  calculate_harmonic_function();
  calculate_mean_coordinates_function(V, Vc, m_weight);
  calculate_selection_threshold();

  Eigen::VectorXd max(Vc.colwise().maxCoeff());
  Eigen::VectorXd min(Vc.colwise().minCoeff());
  Eigen::VectorXd offset = (max-min)*0.1;
  Eigen::MatrixXd V_square(Vc.rows()+4,2);
  for (int i = 0; i < Vc.rows(); ++i)
    V_square.row(i) << Vc.row(i);
  V_square.row(Vc.rows()) << max(0)+offset(0), max(1)+offset(1);
  V_square.row(Vc.rows()+1) << max(0)+offset(0), min(1)-offset(1);
  V_square.row(Vc.rows()+2) << min(0)-offset(0), min(1)-offset(1);
  V_square.row(Vc.rows()+3) << min(0)-offset(0), max(1)+offset(1);
  Eigen::MatrixXi E(4,2);
  E << Vc.rows(),Vc.rows()+1,
       Vc.rows()+1,Vc.rows()+2,
       Vc.rows()+2,Vc.rows()+3,
       Vc.rows()+3,Vc.rows();
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F2;
  igl::triangle::triangulate(V_square, E, H, "a50q", V2, F2);
  m_weight_in_cage.resize(Vc.rows(), V2.rows());
  m_weight_in_cage.setZero();
  Eigen::VectorXi rows = Eigen::VectorXi::LinSpaced(V2.rows()-Vc.rows(), Vc.rows(), V2.rows()-1);
  Eigen::MatrixXd tmp_v, tmp_weight;
  igl::slice(V2, rows, 1, tmp_v);
  calculate_mean_coordinates_function(tmp_v, Vc, tmp_weight);
  igl::slice_into(tmp_weight, rows, 2, m_weight_in_cage);
  for (int i = 0; i < Vc.rows(); ++i)
    m_weight_in_cage(i,i)=1;
  
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  original_mesh = viewer.append_mesh(true);
  h_weight_mesh = viewer.append_mesh(true);
  m_weight_mesh = viewer.append_mesh(true);
  viewer.callback_init = [&](igl::opengl::glfw::Viewer &)
  {
    viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
    left_view = viewer.core_list[0].id;
    right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
    viewer.data(original_mesh).set_visible(false, right_view);
    viewer.data(h_weight_mesh).set_visible(false, left_view);
    viewer.data(m_weight_mesh).set_visible(false, left_view);
    viewer.data(m_weight_mesh).set_visible(false, right_view);
    return true;
  };
  viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) 
  {
    // cout << "resize" << endl;
    v.core(left_view).viewport = Eigen::Vector4f(0, 0, w / 2, h);
    v.core(right_view).viewport = Eigen::Vector4f(w / 2, 0, w - (w / 2), h);
    return true;
  };
  
  set_original_mesh(viewer);
  viewer.data(h_weight_mesh).set_mesh(Vt, Ft);
  viewer.data(h_weight_mesh).set_data(h_weight_in_cage.row(current_cage_index));
  viewer.data(h_weight_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(h_weight_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  }
  viewer.data(h_weight_mesh).add_points(Vi, Eigen::RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(h_weight_mesh).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      Eigen::RowVector3d(0,1,0)
    );
  }

  viewer.data(m_weight_mesh).set_mesh(V2,F2);
  viewer.data(m_weight_mesh).set_data(m_weight_in_cage.row(current_cage_index));
  // viewer.data(m_weight_mesh).set_mesh(V, F);
  // viewer.data(m_weight_mesh).set_data(m_weight.row(current_cage_index));
  viewer.data(m_weight_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(m_weight_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  } 

  viewer.data(h_weight_mesh).show_lines = false;
  viewer.data(m_weight_mesh).show_lines = false;
  
  viewer.callback_key_down = callback_key_down;
  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;


  viewer.launch();
}
