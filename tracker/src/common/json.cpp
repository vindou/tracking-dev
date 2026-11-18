#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace trk {

struct JsonParser {
    const std::string& s;
    std::size_t i = 0;
    std::string err;
    explicit JsonParser(const std::string& t) : s(t) {}
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }
    bool fail(const char* what) {
        if (err.empty()) err = std::string(what) + " at offset " + std::to_string(i);
        return false;
    }
    bool lit(const char* w) {
        const std::string t(w);
        if (s.compare(i, t.size(), t) != 0) return fail("unexpected token");
        i += t.size();
        return true;
    }
    bool str(std::string& out) {
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        i++;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c == '\\') {
                if (i >= s.size()) return fail("bad escape");
                char e = s[i++];
                if (e == 'n') c = '\n';
                else if (e == 't') c = '\t';
                else if (e == 'r') c = '\r';
                else if (e == 'b') c = '\b';
                else if (e == 'f') c = '\f';
                else if (e == 'u') {
                    if (i + 4 > s.size()) return fail("bad \\u");
                    c = char(std::strtol(s.substr(i, 4).c_str(), nullptr, 16) & 0x7F);
                    i += 4;
                } else c = e;
            }
            out.push_back(c);
        }
        if (i >= s.size()) return fail("unterminated string");
        i++;
        return true;
    }
    bool value(Json& v) {
        ws();
        if (i >= s.size()) return fail("unexpected end");
        const char c = s[i];
        if (c == '{') {
            i++;
            v.type_ = Json::Object;
            ws();
            if (i < s.size() && s[i] == '}') { i++; return true; }
            for (;;) {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return fail("expected ':'");
                i++;
                Json e;
                if (!value(e)) return false;
                v.obj_[k] = std::move(e);
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == '}') { i++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            i++;
            v.type_ = Json::Array;
            ws();
            if (i < s.size() && s[i] == ']') { i++; return true; }
            for (;;) {
                Json e;
                if (!value(e)) return false;
                v.arr_.push_back(std::move(e));
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == ']') { i++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.type_ = Json::String; return str(v.str_); }
        if (c == 't') { v.type_ = Json::Bool; v.b_ = true; return lit("true"); }
        if (c == 'f') { v.type_ = Json::Bool; v.b_ = false; return lit("false"); }
        if (c == 'n') { v.type_ = Json::Null; return lit("null"); }
        char* end = nullptr;
        const double d = std::strtod(s.c_str() + i, &end);
        if (end == s.c_str() + i) return fail("bad value");
        i = std::size_t(end - s.c_str());
        v.type_ = Json::Number;
        v.num_ = d;
        return true;
    }
};

Json Json::parse(const std::string& text, std::string& err) {
    JsonParser p(text);
    Json v;
    if (!p.value(v)) { err = p.err; return Json(); }
    p.ws();
    if (p.i != text.size()) { err = "trailing characters at offset " + std::to_string(p.i); return Json(); }
    return v;
}

Json Json::parse_file(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot read " + path; return Json(); }
    std::stringstream ss;
    ss << f.rdbuf();
    Json j = parse(ss.str(), err);
    if (!err.empty()) err = path + ": " + err;
    return j;
}

const Json& Json::operator[](const std::string& key) const {
    static const Json null;
    if (type_ != Object) return null;
    auto it = obj_.find(key);
    return it == obj_.end() ? null : it->second;
}

const Json& Json::operator[](std::size_t i) const {
    static const Json null;
    return type_ == Array && i < arr_.size() ? arr_[i] : null;
}

std::string Json::dump(int indent, int level) const {
    std::string pad(std::size_t(indent * (level + 1)), ' '), pad0(std::size_t(indent * level), ' ');
    std::ostringstream o;
    switch (type_) {
        case Null: o << "null"; break;
        case Bool: o << (b_ ? "true" : "false"); break;
        case Number: {
            char b[40];
            if (std::isfinite(num_) && num_ == std::floor(num_) && std::fabs(num_) < 1e15) std::snprintf(b, sizeof b, "%.0f", num_);
            else std::snprintf(b, sizeof b, "%.17g", num_);
            o << b;
            break;
        }
        case String: {
            o << '"';
            for (char c : str_) {
                if (c == '"' || c == '\\') o << '\\' << c;
                else if (c == '\n') o << "\\n";
                else o << c;
            }
            o << '"';
            break;
        }
        case Array: {
            bool simple = true;
            for (const auto& e : arr_) simple = simple && (e.type_ == Number || e.type_ == Bool || e.type_ == Null);
            o << '[';
            for (std::size_t k = 0; k < arr_.size(); k++) {
                if (!simple) o << '\n' << pad;
                o << arr_[k].dump(indent, level + 1);
                if (k + 1 < arr_.size()) o << (simple ? ", " : ",");
            }
            if (!simple && !arr_.empty()) o << '\n' << pad0;
            o << ']';
            break;
        }
        case Object: {
            o << '{';
            std::size_t k = 0;
            for (const auto& kv : obj_) {
                o << '\n' << pad << '"' << kv.first << "\": " << kv.second.dump(indent, level + 1);
                if (++k < obj_.size()) o << ',';
            }
            if (!obj_.empty()) o << '\n' << pad0;
            o << '}';
            break;
        }
    }
    return o.str();
}

}  // namespace trk
