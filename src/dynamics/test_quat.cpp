#include <Eigen/Geometry>
#include <iostream>
using namespace Eigen;
using namespace std;

template<typename Derived>
inline Quaterniond toQuat(const DenseBase<Derived> &v) {
  assert(v.size() == 4);
  return Quaterniond(v(3), v(0), v(1), v(2));
}

inline Vector4d quatToVec(const Quaterniond &q) {
  return Vector4d(q.x(), q.y(), q.z(), q.w());
}

inline void printQuat(const Quaterniond &q) {
  cout << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w();
}

int main() {
  Quaterniond q(1, 2, 3, 4);
  Vector4d v(2, 3, 4, 1);
  cout << q.coeffs().transpose() << " should equal " << v.transpose() << endl;
  printQuat(Quaterniond(v)); cout << " should equal "; printQuat(q); cout << endl;
  printQuat(toQuat(v)); cout << " should equal "; printQuat(Quaterniond(v)); cout << endl;
  cout << quatToVec(q).transpose() << " should equal " << q.coeffs().transpose() << endl;
  return 0;
}
