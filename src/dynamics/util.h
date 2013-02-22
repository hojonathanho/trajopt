#pragma once

namespace trajopt {
namespace dynamics {

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;

typedef Eigen::Matrix<double, 1, 1> Vector1d;
inline Vector1d makeVector1d(double x) { Vector1d v; v(0) = x; return v; }

inline OR::Vector toOR(const Vector3d& v) {
  return OR::Vector(v(0), v(1), v(2));
}

inline OR::Vector toOR(const Quaterniond &q) {
  return OR::Vector(q.w(), q.x(), q.y(), q.z());
}

inline OR::Transform toOR(const Vector3d &x, const Quaterniond &q) {
  return OR::Transform(toOR(q), toOR(x));
}

template<typename Derived>
inline Quaterniond toQuat(const DenseBase<Derived> &v) {
  assert(v.size() == 4);
  return Quaterniond(v(3), v(0), v(1), v(2));
}

inline Vector4d quatToVec(const Quaterniond &q) {
  return Vector4d(q.x(), q.y(), q.z(), q.w());
}

inline Quaterniond propagatorQuat(const Vector3d& w, double dt) {
  // see http://www.lce.hut.fi/~ssarkka/pub/quat.pdf
  double normw = w.norm();
  Quaterniond phiq;
  if (normw > 0) {
    phiq.vec() = (w/normw)*sin(normw*dt/2);
  } else {
    phiq.vec() = Vector3d::Zero();
  }
  phiq.w() = cos(normw * dt/2);
  return phiq;
}

inline double positivePart(double x) {
  return x > 0 ? x : 0;
}

inline int readIntoMatrix(const VectorXd &vals, MatrixXd &out, int k) {
  for (int i = 0; i < out.rows(); ++i) {
    for (int j = 0; j < out.cols(); ++j) {
      out(i,j) = vals[k++];
    }
  }
  return k;
}

template<class T>
inline vector<T> concat(const vector<T> &a, const vector<T> &b) {
  vector<T> v = a;
  v.insert(v.end(), b.begin(), b.end());
  return v;
}

inline void varArrayIntoVector(const VarArray &a, VarVector &out) {
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      out.push_back(a(i,j));
    }
  }
}

class ZeroCost : public Cost {
  ConvexObjectivePtr convex(const DblVec&, Model* model) {
    ConvexObjectivePtr out(new ConvexObjective(model));
    out->addAffExpr(AffExpr());
    return out;
  }
  double value(const DblVec&) { return 0.; }
};

} // namespace dynamics
} // namespace trajopt
