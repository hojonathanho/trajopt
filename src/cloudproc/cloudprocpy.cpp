#include <boost/python.hpp>
#include <stdexcept>
#include <boost/python/exception_translator.hpp>
#include <boost/foreach.hpp>
#include "cloudproc/cloudproc.hpp"
#include <iostream>
#include "utils/stl_to_string.hpp"
 #include "cloudgrabber.hpp"
#include "cloudproc/mesh_simplification.hpp"
#include <pcl/ros/conversions.h>
#include <pcl/point_types.h>
#include <boost/format.hpp>
#include "hacd_interface.hpp"

using namespace Eigen;
using namespace pcl;
using namespace cloudproc;
using namespace std;
using namespace util;
namespace py = boost::python;
using boost::shared_ptr;
typedef PointCloud<PointXYZ> CloudXYZ;
typedef PointCloud<PointNormal> CloudXYZN;

py::object np_mod, main, globals;

template<typename T>
struct type_traits {
  static const char* npname;
};
template<> const char* type_traits<float>::npname = "float32";
template<> const char* type_traits<int>::npname = "int32";


template <typename T>
T* getPointer(const py::object& arr) {
  long int i = py::extract<long int>(arr.attr("ctypes").attr("data"));
  T* p = (T*)i;
  return p;
}

template<typename T>
py::object toNdarray1(const T* data, size_t dim0) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0), type_traits<T>::npname);
  int* p = getPointer<T>(out);
  memcpy(p, data, dim0*sizeof(T));
  return out;
}
template<typename T>
py::object toNdarray2(const T* data, size_t dim0, size_t dim1) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1), type_traits<T>::npname);
  float* pout = getPointer<float>(out);
  memcpy(pout, data, dim0*dim1*sizeof(float));
  return out;
}
template<typename T>
py::object toNdarray3(const T* data, size_t dim0, size_t dim1, size_t dim2) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1, dim2), type_traits<T>::npname);
  float* pout = getPointer<float>(out);
  memcpy(pout, data, dim0*dim1*dim2*sizeof(float));
  return out;
}
template <class T>
struct PyCloud {
  typedef PointCloud<T> PointCloudT;
  static py::object to2dArray(PointCloudT* cloud) {
    return toNdarray2<float>((const float*)cloud->points.data(), cloud->size(), sizeof(T)/sizeof(float));
  }
  static py::object to3dArray(PointCloudT* cloud) {
    return toNdarray3<float>((const float*)cloud->points.data(), cloud->height, cloud->width, sizeof(T)/sizeof(float));
  }
  static void from2dArray(PointCloudT* cloud, py::object arr) {
    py::object np_mod = py::import("numpy");
    arr = np_mod.attr("array")(arr, "float32");
    int npoints = py::extract<int>(arr.attr("shape")[0]);
    int floatfields = py::extract<int>(arr.attr("shape")[1]);
    FAIL_IF_FALSE(floatfields == sizeof(T)/4);
    cloud->resize(npoints);
    float* p = getPointer<float>(arr);
    memcpy(&cloud->points[0], p,  npoints*floatfields*sizeof(float));
  }
  static void from3dArray(PointCloudT* cloud, py::object arr) {
    py::object np_mod = py::import("numpy");
    arr = np_mod.attr("array")(arr, "float32");
    int height = py::extract<int>(arr.attr("shape")[0]);
    int width = py::extract<int>(arr.attr("shape")[1]);
    int floatfields = py::extract<int>(arr.attr("shape")[2]);
    FAIL_IF_FALSE(floatfields==sizeof(T)/4);
    cloud->resize(width*height);
    cloud->height = height;
    cloud->width = width;
    float* p = getPointer<float>(arr);
    memcpy(&cloud->points[0], p,  width*height*floatfields*sizeof(float));
  }
  static void save(PointCloudT* cloud, const std::string& fname) {
    saveCloud(*cloud, fname);
  }
  static int width(PointCloudT* cloud) {return cloud->width;}
  static int height(PointCloudT* cloud){return cloud->height;}
  static int size(PointCloudT* cloud){return cloud->size();}
};

CloudXYZ::Ptr PolygonMesh_getCloud(const PolygonMesh* mesh) {
  CloudXYZ::Ptr cloud(new CloudXYZ());
  pcl::fromROSMsg(mesh->cloud, *cloud);
  return cloud;
}
py::object PolygonMesh_getVertices(const PolygonMesh* mesh) {
  py::object cloud(PolygonMesh_getCloud(mesh));

  globals["cloud"] = cloud;
  return py::eval("cloud.to2dArray()[:,:3]", globals);
}
py::object PolygonMesh_getFaces(const PolygonMesh* mesh) {
  py::list out;
  BOOST_FOREACH(const pcl::Vertices& poly, mesh->polygons) {
    out.append(toNdarray1<int>((int*)poly.vertices.data(), poly.vertices.size()));
  }
  return out;
}
void PolygonMesh_save(const PolygonMesh* mesh, const std::string& outfile) {
  saveMesh(*mesh, outfile);
}

template<typename T>
void boost_register_cloud_type(const string& pyname) {
  typedef PyCloud<T> PyCloudT;
  typedef PointCloud<T> PointCloudT;
  py::class_<PointCloudT, shared_ptr<PointCloudT> >(pyname.c_str())
//    .def(py::init< shared_ptr<PointCloudT> >() )
    .def("width", &PyCloudT::width)
    .def("height", &PyCloudT::height)
    .def("size", &PyCloudT::size)
    .def("to2dArray", &PyCloudT::to2dArray)
    .def("to3dArray", &PyCloudT::to3dArray)
    .def("from2dArray", &PyCloudT::from2dArray)
    .def("from3dArray", &PyCloudT::from3dArray)
    .def("save", &PyCloudT::save)
    ;
  py::implicitly_convertible< boost::shared_ptr<CloudXYZ>, boost::shared_ptr<CloudXYZ const> >();
}

py::object pyConvexDecompHACD(const PolygonMesh& mesh) {
  vector<PolygonMesh::Ptr> convexmeshes = ConvexDecompHACD(mesh);
  py::list out;
  BOOST_FOREACH(const PolygonMesh::Ptr& convexmesh, convexmeshes) {
    out.append(convexmesh);
  }
  return out;
}


BOOST_PYTHON_MODULE(cloudprocpy) {

  np_mod = py::import("numpy");
  main = py::import("__main__");
  globals = main.attr("__dict__");

  boost_register_cloud_type<PointXYZ>("CloudXYZ");
  boost_register_cloud_type<PointXYZRGB>("CloudXYZRGB");
  boost_register_cloud_type<PointNormal>("CloudXYZN");

  py::def("readPCDXYZ", &readPCD<PointXYZ>);
  py::def("downsampleCloud", &downsampleCloud<PointXYZ>);
  py::def("boxFilter", &boxFilter<PointXYZ>);
  py::def("boxFilterNegative", &boxFilterNegative<PointXYZ>);
  py::def("medianFilter", &medianFilter<PointXYZ>);
  py::def("fastBilateralFilter", &fastBilateralFilter<PointXYZ>);

  py::class_<pcl::PolygonMesh, shared_ptr<pcl::PolygonMesh> >("PolygonMesh")
      .def("getCloud", &PolygonMesh_getCloud)
      .def("getVertices", &PolygonMesh_getVertices)
      .def("getFaces", &PolygonMesh_getFaces)
      .def("save", &PolygonMesh_save)
      ;

  py::def("meshGP3", &meshGP3);
  py::def("meshOFM", &meshOFM);
  py::def("mlsAddNormals", &mlsAddNormals);
  py::def("loadMesh", &loadMesh);
  py::def("quadricSimplifyVTK", &quadricSimplifyVTK);

  py::class_<CloudGrabber, shared_ptr<CloudGrabber> >("CloudGrabber")
      .def("startXYZ", &CloudGrabber::startXYZ)
      .def("getXYZ", &CloudGrabber::getXYZ)
      .def("startXYZRGB", &CloudGrabber::startXYZRGB)
      .def("getXYZRGB", &CloudGrabber::getXYZRGB)
      .def("stop", &CloudGrabber::stop)
      ;

  py::def("convexDecompHACD", &pyConvexDecompHACD);

}
