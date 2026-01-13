#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <stack>
#include <algorithm>

enum NodeType { VAR, NOT, AND, OR, XOR, GROUP };

struct Node {
    NodeType type;
    std::string name;
    std::shared_ptr<Node> left, right;
    Node(NodeType t, const std::string& n="") : type(t), name(n) {}
};

int precedence(char op) {
    if (op == '!') return 3;
    if (op == '&') return 2;
    if (op == '^') return 1;
    if (op == '|') return 0;
    return -1;
}

bool isVar(std::shared_ptr<Node> n) {
    return n && n->type == VAR;
}

std::shared_ptr<Node> parseExpression(const std::string& expr) {
    std::stack<std::shared_ptr<Node>> values;
    std::stack<char> ops;
    std::stack<bool> inGroup;

    auto apply = [&]() {
        char op = ops.top(); ops.pop();
        if (op == '!') {
            auto n = std::make_shared<Node>(NOT);
            n->left = values.top(); values.pop();
            values.push(n);
        } else {
            auto r = values.top(); values.pop();
            auto l = values.top(); values.pop();
            NodeType t = (op=='&'?AND:(op=='|'?OR:XOR));
            auto n = std::make_shared<Node>(t);
            n->left = l; n->right = r;
            values.push(n);
        }
    };

    auto closeGroup = [&]() {
        auto inner = values.top();
        values.pop();

        auto groupNode = std::make_shared<Node>(GROUP);
        groupNode->left = inner;
        values.push(groupNode);
    };

    for (size_t i=0;i<expr.size();++i) {
        char c = expr[i];
        if (isspace(c)) continue;

        if (isalnum(c)) {
            values.push(std::make_shared<Node>(VAR,std::string(1,c)));
        }
        else if (c=='(') {
            ops.push(c);
            inGroup.push(true);
        }
        else if (c==')') {
            while(!ops.empty() && ops.top()!='(') apply();
            ops.pop();

            if (!inGroup.empty() && inGroup.top()) {
                inGroup.pop();
                closeGroup();
            }
        } else {
            while(!ops.empty() && precedence(ops.top())>=precedence(c)) apply();
            ops.push(c);
        }
    }

    while(!ops.empty()) apply();
    return values.top();
}

void collectVars(std::shared_ptr<Node> n, std::vector<std::string>& v) {
    if(!n) return;
    if(n->type==VAR && std::find(v.begin(),v.end(),n->name)==v.end())
        v.push_back(n->name);
    collectVars(n->left,v);
    collectVars(n->right,v);
}

struct Box {
    int x, y, w, h;
};

void flattenSameOps(std::shared_ptr<Node> n, NodeType type, std::vector<std::shared_ptr<Node>>& out) {
    if (!n) return;

    if (n->type == GROUP) {
        out.push_back(n);
        return;
    }

    if (n->type == type) {
        flattenSameOps(n->left, type, out);
        flattenSameOps(n->right, type, out);
    } else {
        out.push_back(n);
    }
}

struct VarPosition {
    std::string name;
    int y;
    int x_right;
};

Box drawExpr(
        cv::Mat& img,
        std::shared_ptr<Node> n,
        std::map<std::string,int>& originalVarY,
        std::map<std::string, VarPosition>& varPositions,
        int x,
        int width,
        int blockH,
        int startY,
        bool drawNotCircle = true,
        double scale = 0.6
) {
    if (n->type == VAR) {
        int y = startY + blockH / 2;

        int tw = cv::getTextSize(n->name, cv::FONT_HERSHEY_SIMPLEX, scale, 1, nullptr).width;
        int px = x + width - tw - 5;
        if (px < x + 5) px = x + 5;

        varPositions[n->name] = {n->name, y, x + width};

        cv::putText(img, n->name, {px, startY + blockH / 2 + 5},
                    cv::FONT_HERSHEY_SIMPLEX, scale, {255, 255, 255}, 1);

        return {x, startY, width, blockH};
    }

    if (n->type == GROUP) {
        return drawExpr(img, n->left, originalVarY, varPositions, x, width, blockH, startY, true, scale);
    }

    if (n->type == NOT) {
        if (isVar(n->left)) {
            int y = startY + blockH / 2;
            int cx = x + width + 10;
            if (drawNotCircle) {
                cv::circle(img, {cx, y}, 8, {255, 255, 255}, 1);
            }
            Box b = drawExpr(img, n->left, originalVarY, varPositions, x, width, blockH, startY, true, scale);
            return {b.x, b.y, b.w + (drawNotCircle ? 30 : 0), b.h};
        }
        Box inner = drawExpr(img, n->left, originalVarY, varPositions, x + 10, width - 10, blockH, startY, true, scale);
        int cy = inner.y + inner.h / 2;
        int cx = x + width + 20;
        if (drawNotCircle) {
            cv::circle(img, {cx, cy}, 6, {255, 255, 255}, 1);
        }
        return {x, inner.y, inner.w + (drawNotCircle ? 30 : 0), inner.h};
    }

    std::vector<std::shared_ptr<Node>> operands;
    flattenSameOps(n, n->type, operands);

    if (operands.size() == 1) {
        return drawExpr(img, operands[0], originalVarY, varPositions, x, width, blockH, startY, true, scale);
    }

    int maxTextWidth = 0;
    for (auto& op : operands) {
        std::vector<std::string> localVars;
        collectVars(op, localVars);
        for (auto& v : localVars) {
            int tw = cv::getTextSize(v, cv::FONT_HERSHEY_SIMPLEX, scale, 1, nullptr).width;
            maxTextWidth = std::max(maxTextWidth, tw);
        }
    }

    int indent = std::max(50, maxTextWidth + 20);

    int curY = startY;
    int minY = startY;
    int maxY = startY;

    for (auto& op : operands) {
        Box b = drawExpr(img, op, originalVarY, varPositions, x + indent, width - indent, blockH, curY, true, scale);
        curY = b.y + b.h;
        maxY = b.y + b.h;
    }

    cv::rectangle(img, {x, minY, width, maxY - minY}, {241, 97, 52}, 2);

    std::string op;
    if (n->type == AND) op = "&";
    else if (n->type == OR) op = "|";
    else if (n->type == XOR) op = "^";

    cv::putText(img, op, {x + 8, minY + 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 1);

    return {x, minY, width, maxY - minY};
}

int main() {
    std::string expr = "!((!A&!B)&((C&!D)|(!E&!F)|((!G&(H^I))|(J&K)|(!L|!M|!N|!O|!P)))&((Q|!R)&(!S&!T&!U)&!V))";
    auto root = parseExpression(expr);

    std::vector<std::string> vars;
    collectVars(root,vars);
    std::sort(vars.begin(),vars.end());

    int stepY = 40, blockH = 30;
    std::map<std::string,int> originalVarY;
    for(int i=0;i<vars.size();++i)
        originalVarY[vars[i]] = 60 + i*stepY;

    cv::Mat img(1200,1800,CV_8UC3,{0,0,0});

    int topY = originalVarY[vars.front()];
    int x0 = 60;
    int globalW = 420;

    std::map<std::string, VarPosition> varPositions;

    drawExpr(img, root, originalVarY, varPositions, x0 + 10, globalW - 20, blockH, topY, false);

    for(int i = 0; i < vars.size(); ++i) {
        std::string varName = vars[i];
        if (varPositions.find(varName) != varPositions.end()) {
            int y = varPositions[varName].y;
            int blockRightX = varPositions[varName].x_right;

            cv::putText(img, std::to_string(i+1),
                        {blockRightX + 40, y + 5},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);

            cv::line(img, {blockRightX + 2, y}, {blockRightX + 40, y}, {255, 255, 255}, 1);
        }
    }

    int minY = 10000, maxY = 0;
    for (auto& var : varPositions) {
        minY = std::min(minY, var.second.y - blockH/2);
        maxY = std::max(maxY, var.second.y + blockH/2);
    }

    int midY = (minY + maxY) / 2;
    cv::line(img, {x0 + 8, midY}, {x0 - 15, midY}, {255, 255, 255}, 1);
    cv::putText(img, std::to_string(vars.size() + 1), {x0 - 40, midY + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
    std::string label = "Y";
    int labelX = x0 + 30;
    int labelY = midY + 6;
    cv::putText(img, label, {labelX, labelY}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);

    if(root->type == NOT)
        cv::circle(img, {x0 + 10, midY}, 8, {255, 255, 255}, 1);

    cv::imshow("UGO", img);
    cv::imwrite("ugo.png", img);
    cv::waitKey(0);

    return 0;
}