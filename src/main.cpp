#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <igl/readOFF.h>
#include <igl//boundary_loop.h>
#include <igl/slice.h>
#include <igl/triangle/triangulate.h>

using namespace std;

Eigen::MatrixXd V;
Eigen::MatrixXi F;

bool callback_key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifiers)
{
  bool handled = false;
  
  return handled;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage harmonic-coordinates mesh.off" << std::endl;
    exit(0);
  }
  igl::readOFF(argv[1],V,F);
  assert(V.rows() > 0);
  std::vector<std::vector<int> > L;
  igl::boundary_loop(F, L);
  // cout << L.size() << endl;
  int* ptr = &L[0][0];
  Eigen::Map<Eigen::VectorXi> index(ptr, L[0].size());
  // cout << index << endl;
  Eigen::MatrixXd new_V;
  igl::slice(V, index, 1, new_V);
  Eigen::MatrixXi new_F;
  new_F.resize(L[0].size(), 2);
  for (int i = 0; i < L[0].size(); ++i) 
  {
    new_F(i, 0) = i;
    new_F(i, 1) = (i+1)%(L[0].size());
  }
  // cout << index << endl;
  // cout << new_F << endl;
  // cout << new_V.rows() << endl;
  // cout << new_V.cols() << endl;
  Eigen::MatrixXi H;
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F2;
  // cout << new_V << endl;
  new_V.conservativeResize(new_V.rows(), 2);
  // cout << new_V << endl;
  igl::triangle::triangulate(new_V, new_F, H, "q", V2, F2);

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.callback_key_down = callback_key_down;
  viewer.data().set_mesh(V2, F2);
  viewer.data().set_face_based(true);

  viewer.data().add_points(new_V, Eigen::RowVector3d(1,0,0));

  viewer.launch();
}
