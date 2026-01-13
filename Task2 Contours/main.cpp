
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <limits>
#include <chrono>
#include <iostream>
#include <windows.h>
#include <cmath>
#include <string>

struct PQItem {
    double dist;
    int idx;
    bool operator<(PQItem const& other) const { return dist > other.dist; }
};

inline int idxAt(int x, int y, int width) { return y * width + x; }
inline void coordFromIdx(int idx, int width, int &x, int &y) { y = idx / width; x = idx % width; }


int main() {
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);

    std::string path = "C:/Users/Vladislav/Desktop/20.png";

    std::cout << "Открытие файла: " << path << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "Ошибка открытия файла: " << path << std::endl;
        return -1;
    }

    cv::Mat bin;
    cv::threshold(gray, bin, 127, 255, cv::THRESH_BINARY);

    int W = bin.cols, H = bin.rows;
    std::cout << "Размер изображения: " << W << " x " << H << std::endl;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    std::cout << "Всего контуров: " << contours.size() << std::endl;
    if (contours.empty()) return 0;

    int external_count = 0, holes_count = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy[i][3] == -1) external_count++;
        else holes_count++;
    }
    std::cout << "Внешние контуры: " << external_count << ", внутренние контуры (отверстия): " << holes_count << std::endl;

    std::vector<std::vector<int>> holesByParent(contours.size());
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        int parent = (i < static_cast<int>(hierarchy.size())) ? hierarchy[i][3] : -1;
        if (parent >= 0 && parent < static_cast<int>(contours.size())) holesByParent[parent].push_back(i);
    }

    cv::Mat result(H, W, CV_8UC3, cv::Scalar(0,0,0));
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        if (hierarchy[i][3] == -1)
            cv::drawContours(result, contours, i, cv::Scalar(255,0,0), 1, cv::LINE_AA);
        else
            cv::drawContours(result, contours, i, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }

    double totalCutLength = 0.0;
    int cutsCount = 0;
    int holesNoAttach = 0;

    const int dx[8] = {1,1,0,-1,-1,-1,0,1};
    const int dy[8] = {0,1,1,1,0,-1,-1,-1};
    const double cost[8] = {1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0), 1.0, std::sqrt(2.0)};

    for (int outerIdx = 0; outerIdx < static_cast<int>(contours.size()); ++outerIdx) {
        if (hierarchy[outerIdx][3] != -1) continue;
        if (holesByParent[outerIdx].empty()) continue;

        cv::Rect roi = cv::boundingRect(contours[outerIdx]);
        for (int holeIdx : holesByParent[outerIdx]) {
            if (holeIdx<0 || holeIdx >= static_cast<int>(contours.size())) continue;
            cv::Rect r = cv::boundingRect(contours[holeIdx]);
            roi |= r;
        }

        int pad = 2;
        int x0 = std::max(0, roi.x - pad);
        int y0 = std::max(0, roi.y - pad);
        int x1 = std::min(W-1, roi.x + roi.width + pad);
        int y1 = std::min(H-1, roi.y + roi.height + pad);
        cv::Rect bigRoi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);

        cv::Mat localMask = cv::Mat::zeros(bigRoi.height, bigRoi.width, CV_8U);

        std::vector<cv::Point> outerShifted;
        outerShifted.reserve(contours[outerIdx].size());
        for (auto &p : contours[outerIdx]) outerShifted.emplace_back(p.x - x0, p.y - y0);
        std::vector<std::vector<cv::Point>> tmp = { outerShifted };
        cv::fillPoly(localMask, tmp, cv::Scalar(255));

        for (int holeIdx : holesByParent[outerIdx]) {
            if (holeIdx<0 || holeIdx >= static_cast<int>(contours.size())) continue;
            std::vector<cv::Point> holeShifted;
            holeShifted.reserve(contours[holeIdx].size());
            for (auto &p : contours[holeIdx]) holeShifted.emplace_back(p.x - x0, p.y - y0);
            std::vector<std::vector<cv::Point>> tmpHole = { holeShifted };
            cv::fillPoly(localMask, tmpHole, cv::Scalar(0));
        }

        if (cv::countNonZero(localMask) == 0) continue;

        int LW = localMask.cols, LH = localMask.rows;
        int Npix = LW * LH;

        const double INF = std::numeric_limits<double>::infinity();
        std::vector<double> dist(static_cast<size_t>(Npix), INF);
        std::vector<int> parent(static_cast<size_t>(Npix), -1);
        std::vector<char> visited(static_cast<size_t>(Npix), 0);
        std::priority_queue<PQItem> pq;

        for (auto &p : outerShifted) {
            int ox = p.x, oy = p.y;
            if (ox < 0 || ox >= LW || oy < 0 || oy >= LH) continue;
            if (localMask.at<uchar>(oy,ox) == 0) continue;
            int id = idxAt(ox, oy, LW);
            if (dist[id] > 0.0) {
                dist[id] = 0.0;
                parent[id] = id;
                pq.push(PQItem{0.0, id});
            }
        }

        while (!pq.empty()) {
            PQItem cur = pq.top(); pq.pop();
            int u = cur.idx;
            if (visited[u]) continue;
            visited[u] = 1;

            int ux, uy; coordFromIdx(u, LW, ux, uy);

            for (int k = 0; k < 8; ++k) {
                int nx = ux + dx[k], ny = uy + dy[k];
                if (nx < 0 || nx >= LW || ny < 0 || ny >= LH) continue;
                int v = idxAt(nx, ny, LW);
                if (visited[v]) continue;
                if (localMask.at<uchar>(ny, nx) == 0) continue;
                double nd = dist[u] + cost[k];
                if (nd < dist[v]) {
                    dist[v] = nd;
                    parent[v] = u;
                    pq.push(PQItem{nd, v});
                }
            }
        }

        for (int holeIdx : holesByParent[outerIdx]) {
            if (holeIdx<0 || holeIdx >= static_cast<int>(contours.size())) continue;
            std::vector<cv::Point> innerShifted;
            innerShifted.reserve(contours[holeIdx].size());
            for (auto &p : contours[holeIdx]) innerShifted.emplace_back(p.x - x0, p.y - y0);

            double bestd = INF;
            int bestId = -1;
            for (size_t ip = 0; ip < innerShifted.size(); ++ip) {
                int ix = innerShifted[ip].x, iy = innerShifted[ip].y;
                if (ix < 0 || ix >= LW || iy < 0 || iy >= LH) continue;

                for (int ny=-1; ny<=1; ++ny) for (int nx=-1; nx<=1; ++nx) {
                        if (nx==0 && ny==0) continue;
                        int mx = ix + nx, my = iy + ny;
                        if (mx < 0 || mx >= LW || my < 0 || my >= LH) continue;
                        if (localMask.at<uchar>(my, mx) == 0) continue;
                        int mid = idxAt(mx, my, LW);
                        if (dist[mid] < bestd) {
                            bestd = dist[mid];
                            bestId = mid;
                        }
                    }
            }

            if (bestId == -1 || bestd == INF) {
                ++holesNoAttach;
                continue;
            }

            std::vector<int> pathIdx;
            int cur = bestId;
            pathIdx.push_back(cur);
            int safety = 0;
            while (parent[cur] != cur && parent[cur] != -1 && safety < LW*LH) {
                cur = parent[cur];
                pathIdx.push_back(cur);
                ++safety;
            }
            if (safety >= LW*LH) {
                ++holesNoAttach;
                continue;
            }

            double pathLen = 0.0;
            for (size_t pi = 0; pi + 1 < pathIdx.size(); ++pi) {
                int a = pathIdx[pi], b = pathIdx[pi+1];
                int ax, ay, bx, by;
                coordFromIdx(a, LW, ax, ay);
                coordFromIdx(b, LW, bx, by);
                cv::Point A(ax + x0, ay + y0), B(bx + x0, by + y0);
                cv::line(result, A, B, cv::Scalar(0,255,0), 1, cv::LINE_AA);
                pathLen += std::sqrt(static_cast<double>((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y)));
            }

            totalCutLength += pathLen;
            ++cutsCount;
        }
    }

    std::string outPath = "C:/Users/Vladislav/Desktop/OpenCV_Task.png";
    bool ok = cv::imwrite(outPath, result);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    std::cout << "Всего разрезов " << cutsCount << std::endl;
    std::cout << "Общая длина разрезов (пиксели): " << totalCutLength << std::endl;
    std::cout << "Заняло времени: " << elapsed_s << " сек" << std::endl;
    std::cout << "Файл сохранён: " << outPath << std::endl;

    return 0;
}

