// Small JSON reader and writer (configuration files; no third-party JSON library is permitted).
// Supports objects, arrays, numbers, strings (with \" \\ \n \t \uXXXX escapes for ASCII), true/false/null.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace trk {

class Json {
public:
    enum Type { Null, Bool, Number, String, Array, Object };
    Json() = default;
    static Json parse(const std::string& text, std::string& err);
    static Json parse_file(const std::string& path, std::string& err);

    Type type() const { return type_; }
    bool is_object() const { return type_ == Object; }
    bool has(const std::string& key) const { return type_ == Object && obj_.count(key) > 0; }
    const Json& operator[](const std::string& key) const;   // Null if missing
    const Json& operator[](std::size_t i) const;            // Null if out of range
    std::size_t size() const { return type_ == Array ? arr_.size() : type_ == Object ? obj_.size() : 0; }
    double num(double def = 0) const { return type_ == Number ? num_ : def; }
    bool boolean(bool def = false) const { return type_ == Bool ? b_ : def; }
    const std::string& str() const { return str_; }
    const std::map<std::string, Json>& items() const { return obj_; }
    const std::vector<Json>& elements() const { return arr_; }

    // Builders (for writing files)
    static Json number(double v) { Json j; j.type_ = Number; j.num_ = v; return j; }
    static Json string(const std::string& s) { Json j; j.type_ = String; j.str_ = s; return j; }
    static Json boolean_value(bool b) { Json j; j.type_ = Bool; j.b_ = b; return j; }
    static Json array() { Json j; j.type_ = Array; return j; }
    static Json object() { Json j; j.type_ = Object; return j; }
    Json& set(const std::string& key, Json v) { type_ = Object; obj_[key] = std::move(v); return *this; }
    Json& push(Json v) { type_ = Array; arr_.push_back(std::move(v)); return *this; }
    std::string dump(int indent = 1, int level = 0) const;

private:
    Type type_ = Null;
    bool b_ = false;
    double num_ = 0;
    std::string str_;
    std::vector<Json> arr_;
    std::map<std::string, Json> obj_;
    friend struct JsonParser;
};

}  // namespace trk
