// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Eigen::Vector3f p = { (float) x, (float) y, 1.0f };
    
    Eigen::Vector3f ab = _v[1] - _v[0];
    Eigen::Vector3f bc = _v[2] - _v[1];
    Eigen::Vector3f ca = _v[0] - _v[2];

    Eigen::Vector3f ap = p - _v[0];
    Eigen::Vector3f bp = p - _v[1];
    Eigen::Vector3f cp = p - _v[2];

    Eigen::Vector3f c0 = ab.cross(ap);
    Eigen::Vector3f c1 = bc.cross(bp);
    Eigen::Vector3f c2 = ca.cross(cp);

    return (c0[2] > 0.0f && c1[2] > 0.0f && c2[2] > 0.0f);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    int xMin, yMin, xMax, yMax;
    xMin = (int) v[0].x();
    yMin = (int) v[0].y();
    xMax = (int) v[0].x();
    yMax = (int) v[0].y();
    for (int i = 1; i < 3; i++) {
        if (v[i].x() < xMin) xMin = v[i].x();
        if (v[i].x() > xMax) xMax = v[i].x();
        if (v[i].y() < yMin) yMin = v[i].y();
        if (v[i].y() > yMax) yMax = v[i].y();
    }
    xMax++;
    yMax++;

    // If so, use the following code to get the interpolated z value.
    // auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    // z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
    bool superSamling = false;
    Eigen::Vector3f point;
    if(~superSamling) {
        for(int i = xMin; i < xMax; i++) {
            for(int j = yMin; j < yMax; j++) {
                // if pixel inside triangle
                if(insideTriangle(i + 0.5, j + 0.5, t.v)) {
                    // interpolate the depth
                    auto[alpha, beta, gamma] = computeBarycentric2D(i + 0.5, j + 0.5, t.v);
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    point.x()= i;
                    point.y()= j;
                    if(z_interpolated < depth_buf[get_index(i, j)]) {
                        set_pixel(point, t.getColor());
                        depth_buf[get_index(i, j)] = z_interpolated;
                    }   
                } 
            }
        }
    } else {
        //SuperSampling data for each pixel
        std::vector<float> pixelDepth(4, 0.0f);
        std::vector<int> pixelCount(4, 0);
        int colorCoef;
        for(int i = xMin - 1; i <= xMax; i++) {
            for(int j = yMin - 1; j <= yMax; j++) {
                std::fill(pixelCount.begin(), pixelCount.end(), 0);
                std::fill(pixelDepth.begin(), pixelDepth.end(), 0.0f);

                colorCoef = 0;
                bool inside = false;
                if(insideTriangle(i + 0.25, j + 0.25, t.v)) {
                    pixelCount[0] = 1;
                    inside = true;
                }
                if(insideTriangle(i + 0.25, j + 0.75, t.v)) {
                    pixelCount[1] = 1;
                    inside = true;
                }
                if(insideTriangle(i + 0.75, j + 0.25, t.v)) {
                    pixelCount[2] = 1;
                    inside = true;
                }
                if(insideTriangle(i + 0.75, j + 0.75, t.v)) {
                    pixelCount[3] = 1;
                    inside = true;
                }
                
                if(inside) {
                    auto[alpha0, beta0, gamma0] = computeBarycentric2D(i + 0.5, j + 0.5, t.v);
                    float w_reciprocal = 1.0 / (alpha0 / v[0].w() + beta0 / v[1].w() + gamma0 / v[2].w());
                    float z_interpolated_a = alpha0 * v[0].z() / v[0].w() + beta0 * v[1].z() / v[1].w() + gamma0 * v[2].z() / v[2].w();
                    z_interpolated_a *= w_reciprocal;

                    for(int m = 0; m < 4; m++) {
                        float z_interpolated;
                        if(m == 0) {
                            auto[alpha1, beta1, gamma1] = computeBarycentric2D(i+0.25, j+0.25, t.v);
                            w_reciprocal = 1.0 / (alpha1 / v[0].w() + beta1 / v[1].w() + gamma1 / v[2].w());
                            z_interpolated = alpha1 * v[0].z() / v[0].w() + beta1 * v[1].z() / v[1].w() + gamma1 * v[2].z() / v[2].w();
                        } else if (m == 1) {
                            auto[alpha2, beta2, gamma2] = computeBarycentric2D(i+0.25, j+0.75, t.v);
                            w_reciprocal = 1.0 / (alpha2 / v[0].w() + beta2 / v[1].w() + gamma2 / v[2].w());
                            z_interpolated = alpha2 * v[0].z() / v[0].w() + beta2 * v[1].z() / v[1].w() + gamma2 * v[2].z() / v[2].w();
                        } else if (m == 2) {
                            auto[alpha3, beta3, gamma3] = computeBarycentric2D(i+0.75, j+0.25, t.v);
                            w_reciprocal = 1.0 / (alpha3 / v[0].w() + beta3 / v[1].w() + gamma3 / v[2].w());
                            z_interpolated = alpha3 * v[0].z() / v[0].w() + beta3 * v[1].z() / v[1].w() + gamma3 * v[2].z() / v[2].w();
                        } else {
                            auto[alpha4, beta4, gamma4] = computeBarycentric2D(i+0.75, j+0.75, t.v);
                            w_reciprocal = 1.0 / (alpha4 / v[0].w() + beta4 / v[1].w() + gamma4 / v[2].w());
                            z_interpolated = alpha4 * v[0].z() / v[0].w() + beta4 * v[1].z() / v[1].w() + gamma4 * v[2].z() / v[2].w();
                        }
                        pixelDepth[m] = z_interpolated * w_reciprocal;
                    }
                    int minDepth = pixelDepth[0];
                    for(int k = 0; k < 4; k++) {
                        if(pixelDepth[k] < depth_buf[get_index(i, j)] && pixelCount[k] == 1)
                            colorCoef = colorCoef + pixelCount[k];
                        if(pixelDepth[k] < minDepth) minDepth = pixelDepth[k];
                    }
                    point.x()= i;
                    point.y()= j;
                    if(minDepth < depth_buf[get_index(i, j)]) {
                        set_pixel(point, (t.getColor() * colorCoef) / 4 + (4 - colorCoef) * frame_buf[(height - 1 - j) * width + i] / 4);
                        depth_buf[get_index(i, j)] = minDepth;
                    }
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on