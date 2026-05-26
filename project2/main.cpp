// clang++ -std=c++11 royal.cpp lbfgs.c -o fluid
// ./fluid
// ffmpeg -framerate 30 -i test%d.png -c:v libx264 -pix_fmt yuv420p fluid.mp4

#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <sstream>

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "lbfgs.h"


#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#include <random>
#include <cstring>
#include <cstdio>
#ifndef M_PI
#endif

double sqr(double x) { return x * x; };

//
static std::default_random_engine engine(0);
static std::uniform_real_distribution<double> uniform01(0.0, 1.0);
//

class Vector {
public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}


class Polygon {
public:
    double area() const {
        if (vertices.size()<3) {
            return 0;
        }
        double A =0;
        for (int i = 0; i < (int)vertices.size(); i++) {
            const Vector& P = vertices[i];
            const Vector& Q = vertices[(i+1)%vertices.size()];
            A+=P[0]*Q[1]-P[1]*Q[0];
        }
        return std::abs(0.5 * A);
    }


    Vector centroid() const {
        if (vertices.size()<3) {
            return Vector(0, 0);
        }
        double A =0;
        Vector C(0, 0);
        for (int i = 0; i < (int)vertices.size(); i++) {
            const Vector& P =vertices[i];
            const Vector& Q =vertices[(i+1)%vertices.size()];
            double c =P[0]*Q[1]-P[1]*Q[0];
            A +=c;
            C =C+(P+Q)*c;
        }
        if (std::abs(A) < 1e-14) {
            return Vector(0, 0);
        }
        return C/(3.0*A);
    }




    double integral_square_distance(const Vector& Pi) {
        if (vertices.size()<3) {
            return 0;
        }
        double a = 0;
        for (int i = 0; i < (int)vertices.size(); i++) {
            const Vector& A = vertices[i];
            const Vector& B = vertices[(i + 1) % vertices.size()];
            a += (A[0]*B[1]-A[1]*B[0]) * ((A.norm2() + dot(A, B) + B.norm2()) / 12.0 - dot(Pi, A + B) / 3.0 + Pi.norm2() / 2.0);
        }
        return std::abs(a);
    }

    std::vector<Vector> vertices;
};


void save_frame(const std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

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
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}


class VoronoiDiagram {

    //Tita BX24 walked me through this

public:

    VoronoiDiagram() {
        n_disk = 100;
        unit_disk.resize(n_disk);
        for (int i = 0; i < n_disk; i++) {
            double theta = 2.0*M_PI * i/(double)n_disk;
            unit_disk[i] =Vector(cos(theta), sin(theta));
        }
    };


    void compute() {

        // TODO Lab 1 (Voronoi)
        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (Lab 3, fluids) : also clip it by a disk of radius sqrt(w_i - w_air) centered at Pi

        cells.clear();
        cells.resize(points.size());

        if (weights.empty()) {
            weights.resize(points.size(), 0.0);
        }

        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i <(int)points.size(); i++) {
            Polygon cell;
            cell.vertices.push_back(Vector(0, 0));
            cell.vertices.push_back(Vector(1, 0));
            cell.vertices.push_back(Vector(1, 1));
            cell.vertices.push_back(Vector(0, 1));
            for (int j=0; j<(int)points.size(); j++) {
                if (i == j) continue;
                double wi = (i < (int)weights.size()) ? weights[i] : 0.0;
                double wj = (j < (int)weights.size()) ? weights[j] : 0.0;
                cell =clip_by_bisector(cell, points[i], points[j], wi, wj);
                if (cell.vertices.empty()) {
                    break;
                }
            }
            if ((int)weights.size()==(int)points.size()+1 && !cell.vertices.empty()) {
                double radius = sqrt(std::max(0.0, weights[i]-weights[weights.size()-1]));
                for (int j = 0; j <n_disk; j++) {
                    Vector u=points[i]+radius*unit_disk[j];
                    Vector v=points[i]+radius*unit_disk[(j+1)%n_disk];
                    cell=clip_by_edge(cell, u, v);
                    if (cell.vertices.empty()) {
                        break;
                    }
                }
            }
            cells[i] = cell;
        }
    }

    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {

        // TODO Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk

        Polygon result;
        if (V.vertices.empty()) {
            return result;
        }
        result.vertices.reserve(V.vertices.size()+1);

        for (int i = 0; i<(int)V.vertices.size(); i++) {
            Vector A =V.vertices[i];
            Vector B =V.vertices[(i+1)% V.vertices.size()];
            double da=(v-u)[0]*(A-u)[1]-(v-u)[1]*(A-u)[0];
            double db=(v-u)[0]*(B-u)[1]-(v-u)[1]*(B-u)[0];
            bool Ain = false;
            if (da >= -1e-12) {
                Ain=true;
            }
            bool Bin = false;
            if (db >= -1e-12) {
                Bin=true;
            }
            
            if (Ain && Bin) {
                result.vertices.push_back(B);
            }
            else if (Ain && !Bin) {
                if (std::abs(da-db)>1e-20) {
                    double t=da/(da-db);
                    result.vertices.push_back(A+t*(B-A));
                }
            }
            else if (!Ain && Bin) {
                if (std::abs(da-db)>1e-20) {
                    double t = da/(da-db);
                    result.vertices.push_back(A+t*(B-A));
                }
                result.vertices.push_back(B);
            }
        }

        return result;
    }

    static Polygon clip_by_bisector(const Polygon& V, const Vector& Pi, const Vector& Pj, double w0, double wi) {

        // TODO Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the current site of the Voronoi cell being computed) and Pi (another site)
        
        // TODO Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre cells, i.e., w0 != w1

        Polygon result;
        Vector N = Pj-Pi;
        double rhs =0.5 *(Pj.norm2()- Pi.norm2()+ w0-wi);
        result.vertices.reserve(V.vertices.size()+1);
        for (int i=0; i<V.vertices.size(); i++) {
            Vector A=V.vertices[i];
            Vector B=V.vertices[(i+1)%V.vertices.size()];
            bool Ain = false;
            if (dot(A, N)-rhs<=1e-12) {
                Ain=true;
            }
            bool Bin = false;
            if (dot(B, N)-rhs<=1e-12) {
                Bin=true;
            }


            Vector M= (Pi+Pj)/2+((w0-wi)/(2*N.norm2())) *N;
            // double a= dot(B-A, Pj-Pi);
            double t= dot(M -A, Pj-Pi)/dot(B-A, Pj-Pi);
            Vector P= A+t*(B-A);

            if (Ain && Bin) {
                result.vertices.push_back(B);
            }
            else if (Ain && !Bin) {
                result.vertices.push_back(P);
            }
            else if (!Ain && Bin) {
                result.vertices.push_back(P);
                result.vertices.push_back(B);
            }
        }

        return result;
    }


    std::vector<Vector> points;    // Lab 1 (Voronoi) : the sites to consider

    std::vector<double> weights;   // Lab 2 (OT) : the weight associated to each site (the Laguerre weight, i.e. the dual optimal transport variables to be optimized)
    
    std::vector<Polygon> cells;   // Lab 1 : the polygons representing each individual cell

    //for lab 3
    std::vector<Vector> unit_disk;
    int n_disk;

};

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(
    void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
    int n, int k, int ls) {
    printf("Iteration %d:\n", k);
    printf("fx = %f\n", fx);
    printf("xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    return 0;
}
static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
);

// Lab 2 
class OptimalTransport {

public:
    OptimalTransport() {
        fluid_volume = 0.6;
    };

    void optimize() {
        int ret = 0;
        int N =vor.weights.size();
        lbfgsfloatval_t fx;
        std::vector<double> weights(N, 0.0);
        memcpy(&weights[0], &vor.weights[0], N * sizeof(weights[0]));
        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        ret = lbfgs(N, &weights[0], &fx, evaluate, NULL, (void*)this, &param);
        // lab 3 n+1 change
        memcpy(&vor.weights[0], &weights[0], N * sizeof(weights[0]));
        vor.compute();
    }

    VoronoiDiagram vor;
    double fluid_volume;
};


static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);
    std::memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
    ot->vor.compute();

    

    // Lab 2 (Optimal transport) : compute the function to be minimized (fx) and its gradient (g[i], i=0..n-1)
    // Lab 3 (fluid) : adapt these functions to support partial optimal transport (now "n" has been increased by 1 to account for the air variable)

    lbfgsfloatval_t fx = 0.0;
    // g[i] = ...
    // fx = ...

    // An explained this to me

    bool pot = (n == (int)ot->vor.points.size() + 1);
    int Nfluid =pot ? n-1 : n;
    double dvf= ot->fluid_volume;
    double dva= 1.0-dvf;
    double dpp= dvf/(double)Nfluid;
    double t =0.0;

    for (int i = 0; i < Nfluid; i++) {
        double ai =ot->vor.cells[i].area();
        t+=ai;
        g[i]=dpp-ai;
        fx+=ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);
        fx+=-x[i]*ai;
        fx+=dpp*x[i];
    }
    if (pot) {
        g[n - 1] = dva - (1.0-t);
        fx += x[n - 1] * (dva - (1.0-t));
    }
    return fx;
}




// Lab 3 (fluids)
class Fluid {
public:
    Fluid(int N_particles = 10000) : N_particles(N_particles) {
        fluid_volume = 0.6;
        particles.resize(N_particles);
        velocities.resize(N_particles, Vector(0, 0));
        for (int i = 0; i < N_particles; i++) {
            particles[i] = Vector(0.1+0.8*uniform01(engine), 0.1+0.8*uniform01(engine));
        }
        ot.fluid_volume =fluid_volume;
        ot.vor.points =particles;
        // ot.vor.weights.resize(N_particles+1, 1.0);
        // ot.vor.weights[N_particles] = 0.9;
        ot.vor.weights.resize(N_particles + 1, 0.0);
        ot.vor.weights[N_particles] = -0.001;
        ot.vor.compute();
    }

    // Lab 3 : advance the simulation dt in time
    void time_step(double dt) {
        double epsilon = 0.004;
        Vector g(0, -9.81);
        double m_i = 200.0;

        // TODO Lab 3 : 
        // Compute semi-discrete partial optimal transport
        // for all particles, add gravity and spring force towards cell centroid, integrate acceleration->velocity and velocity->position

        ot.vor.points = particles;
        ot.optimize();

        for (int i = 0; i < N_particles; i++) {
            if (ot.vor.cells[i].vertices.size()<3) {
                continue;
            }
            Vector centroid =ot.vor.cells[i].centroid();
            Vector F_spring =(centroid-particles[i])/(epsilon*epsilon);
            Vector F = F_spring+m_i*g;
            velocities[i] =velocities[i]+(dt/m_i)*F;
            particles[i] =particles[i]+dt*velocities[i];
            if (particles[i][0]<0) {
                particles[i][0]=0;
                velocities[i][0]*=-0.5;
            }
            if (particles[i][0]>1) {
                particles[i][0]=1;
                velocities[i][0]*=-0.5;
            }
            if (particles[i][1]<0) {
                particles[i][1]=0;
                velocities[i][1]*=-0.5;
            }
            if (particles[i][1]>1) {
                particles[i][1]=1;
                velocities[i][1]*=-0.5;
            }
        }
    }

    // just run the full simulation
    void run_simulation() {
        double dt = 0.006; // made it a bit faster
        for (int i = 0; i < 85; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "test", i);
        }
    }

    int N_particles;

    OptimalTransport ot;
    std::vector<Vector> particles;  // the position of all particles
    std::vector<Vector> velocities; // the velocities of all particles
    double fluid_volume; // you decide the fraction of the unit square occupied by the fluid
};

// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon>& polygons, std::string filename, const std::vector<Vector>* points = NULL, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");
        for (int i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n", (*points)[i][0] * 1000., 1000. - (*points)[i][1] * 1000);
        }
        fprintf(f, "</g>\n");

    }

    fprintf(f, "</svg>\n");
    fclose(f);
}




int main() {
    Fluid fluid(100);
    fluid.run_simulation();
    return 0;
}

    
