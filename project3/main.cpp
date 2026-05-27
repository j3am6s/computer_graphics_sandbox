#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <map>
#include <string>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif


double sqr(double x) { return x * x; };

class Vector {
public:
	explicit Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	double norm() const {
		return sqrt(norm2());
	}
	void normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	double operator[](int i) const { return data[i]; };
	double& operator[](int i) { return data[i]; };
	double data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}



// Class used when loading meshes 
class TriangleIndices {
public:
	TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1) {
		vtx[0] = vtxi; vtx[1] = vtxj; vtx[2] = vtxk;
		uv[0] = uvi; uv[1] = uvj; uv[2] = uvk;
		n[0] = ni; n[1] = nj; n[2] = nk;
		this->group = group;
	};
	int vtx[3]; // indices within the vertex coordinates array
	int uv[3];  // indices within the uv coordinates array
	int n[3];   // indices within the normals array
	int group;  // face group
};

class TriangleMesh  {
public:
	TriangleMesh(){};

	// read an .obj file
	void readOBJ(const char* obj) {
		std::ifstream f(obj);
		if (!f) return;

		std::map<std::string, int> mtls;
		int curGroup = -1, maxGroup = -1;

		// OBJ indices are 1-based and can be negative (relative), this normalizes them
		auto resolveIdx = [](int i, int size) {
			return i < 0 ? size + i : i - 1;
		};

		auto setFaceVerts = [&](TriangleIndices& t, int i0, int i1, int i2) {
			t.vtx[0] = resolveIdx(i0, vertices.size());
			t.vtx[1] = resolveIdx(i1, vertices.size());
			t.vtx[2] = resolveIdx(i2, vertices.size());
		};
		auto setFaceUVs = [&](TriangleIndices& t, int j0, int j1, int j2) {
			t.uv[0] = resolveIdx(j0, uvs.size());
			t.uv[1] = resolveIdx(j1, uvs.size());
			t.uv[2] = resolveIdx(j2, uvs.size());
		};
		auto setFaceNormals = [&](TriangleIndices& t, int k0, int k1, int k2) {
			t.n[0] = resolveIdx(k0, normals.size());
			t.n[1] = resolveIdx(k1, normals.size());
			t.n[2] = resolveIdx(k2, normals.size());
		};

		std::string line;
		while (std::getline(f, line)) {
			// Trim trailing whitespace
			line.erase(line.find_last_not_of(" \r\t\n") + 1);
			if (line.empty()) continue;

			const char* s = line.c_str();

			if (line.rfind("usemtl ", 0) == 0) {
				std::string matname = line.substr(7);
				auto result = mtls.emplace(matname, maxGroup + 1);
				if (result.second) {
					curGroup = ++maxGroup;
				} else {
					curGroup = result.first->second;
				}
			} else if (line.rfind("vn ", 0) == 0) {
				Vector v;
				sscanf(s, "vn %lf %lf %lf", &v[0], &v[1], &v[2]);
				normals.push_back(v);
			} else if (line.rfind("vt ", 0) == 0) {
				Vector v;
				sscanf(s, "vt %lf %lf", &v[0], &v[1]);
				uvs.push_back(v);
			} else if (line.rfind("v ", 0) == 0) {
				Vector pos, col;
				if (sscanf(s, "v %lf %lf %lf %lf %lf %lf", &pos[0], &pos[1], &pos[2], &col[0], &col[1], &col[2]) == 6) {
					for (int i = 0; i < 3; i++) col[i] = std::min(1.0, std::max(0.0, col[i]));
					vertexcolors.push_back(col);
				} else {
					sscanf(s, "v %lf %lf %lf", &pos[0], &pos[1], &pos[2]);
				}
				vertices.push_back(pos);
			}
			else if (line[0] == 'f') {
				int i[4], j[4], k[4], offset, nn;
				const char* cur = s + 1;
				TriangleIndices t;
				t.group = curGroup;

				// Try each face format: v/vt/vn, v/vt, v//vn, v
				if ((nn = sscanf(cur, "%d/%d/%d %d/%d/%d %d/%d/%d%n", &i[0], &j[0], &k[0], &i[1], &j[1], &k[1], &i[2], &j[2], &k[2], &offset)) == 9) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d/%d %d/%d %d/%d%n", &i[0], &j[0], &i[1], &j[1], &i[2], &j[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]);
				} else if ((nn = sscanf(cur, "%d//%d %d//%d %d//%d%n", &i[0], &k[0], &i[1], &k[1], &i[2], &k[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d %d %d%n", &i[0], &i[1], &i[2], &offset)) == 3) {
					setFaceVerts(t, i[0], i[1], i[2]);
				}
				else continue;

				indices.push_back(t);
				cur += offset;

				// Fan triangulation for polygon faces (4+ vertices)
				while (*cur && *cur != '\n') {
					TriangleIndices t2;
					t2.group = curGroup;
					if ((nn = sscanf(cur, " %d/%d/%d%n", &i[3], &j[3], &k[3], &offset)) == 3) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d/%d%n", &i[3], &j[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]);
					} else if ((nn = sscanf(cur, " %d//%d%n", &i[3], &k[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d%n", &i[3], &offset)) == 1) {
						setFaceVerts(t2, i[0], i[2], i[3]);
					} else { 
						cur++; 
						continue; 
					}

					indices.push_back(t2);
					cur += offset;
					i[2] = i[3]; j[2] = j[3]; k[2] = k[3];
				}
			}
		}
	}
	

    void Tutte() {
		
		
		uvs.resize(vertices.size());
		
		// TODO : fill the uv coordinates with Tutte embedding
		// locate boundary vertices
		// put them on a unit circle within [0,1]^2
		// then iterate : for each interior vertices, set their parameterization to be the average of their neighbor's parameterization.
		
		
	}
	

	void save_image(std::string filename) {
		constexpr int W = 800, H = 800;
		constexpr double edge_width = 2.0;
		constexpr double edge_width2 = edge_width * edge_width;

		std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

	#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < (int)indices.size(); ++i) {
			Vector V[3] = {uvs[indices[i].vtx[0]], uvs[indices[i].vtx[1]], uvs[indices[i].vtx[2]]};
			
			const int n = 3;
			std::vector<double> xs(n), ys(n);
			double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
			for (int j = 0; j < n; ++j) {
				xs[j] = V[j][0] * W;
				ys[j] = V[j][1] * H;
				xmin = std::min(xmin, xs[j]);
				ymin = std::min(ymin, ys[j]);
				xmax = std::max(xmax, xs[j]);
				ymax = std::max(ymax, ys[j]);
			}

			int x0 = std::max(0, (int)std::floor(xmin - edge_width));
			int y0 = std::max(0, (int)std::floor(ymin - edge_width));
			int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
			int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
			for (int y = y0; y <= y1; ++y) {
				for (int x = x0; x <= x1; ++x) {
					const double px = x + 0.5, py = y + 0.5;

					int prev_sign = 0;
					bool isInside = true;
					bool isEdge = false;

					for (int j = 0; j < n; ++j) {
						int k = (j + 1) % n;

						double ax = xs[j], ay = ys[j];
						double bx = xs[k], by = ys[k];
						double dx = bx - ax, dy = by - ay;
						double qx = px - ax, qy = py - ay;

						double det = qx * dy - qy * dx;
						int s = (det > 1e-12) - (det < -1e-12);

						if (s != 0) {
							if (prev_sign != 0 && s != prev_sign) {
								isInside = false;
								break;
							}
							prev_sign = s;
						}

						double len2 = dx * dx + dy * dy;
						double dot = qx * dx + qy * dy;
						if (dot >= 0.0 && dot <= len2 && det * det <= edge_width2 * len2)
							isEdge = true;
					}

					if (isInside) {
						int id = (H - 1 - y) * W + x;
						inside[id] = 1;
						if (isEdge) edge[id] = 1;
					}
				}
			}
		}

		std::vector<unsigned char> image(W * H * 3, 255);

	#pragma omp parallel for
		for (int i = 0; i < W * H; ++i) {
			if (edge[i]) {
				image[3 * i + 0] = 0;
				image[3 * i + 1] = 0;
				image[3 * i + 2] = 0;
			}
			/*else if (inside[i]) {
				image[3 * i + 0] = 0;
				image[3 * i + 1] = 0;
				image[3 * i + 2] = 255;
			}*/
		}


		stbi_write_png(filename.c_str(), W, H, 3, image.data(), W * 3);
	}



	std::vector<TriangleIndices> indices;
	std::vector<Vector> vertices;
	std::vector<Vector> normals;
	std::vector<Vector> uvs;
	std::vector<Vector> vertexcolors;

};


int main() {
	TriangleMesh goethe;
	goethe.readOBJ("goethe.obj");
    goethe.Tutte();
	goethe.save_image("tutte.png");

	return 0;
}
