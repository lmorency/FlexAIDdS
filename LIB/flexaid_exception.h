#ifndef FLEXAID_EXCEPTION_H
#define FLEXAID_EXCEPTION_H

#include <stdexcept>
#include <string>

class FlexAIDException : public std::runtime_error {
public:
    explicit FlexAIDException(const std::string& message)
        : std::runtime_error(message) {}

    explicit FlexAIDException(const char* message)
        : std::runtime_error(message) {}

    FlexAIDException(const std::string& message, int exit_code)
        : std::runtime_error(message), exit_code_(exit_code) {}

    int exit_code() const noexcept { return exit_code_; }

private:
    int exit_code_ = 1;
};

#endif // FLEXAID_EXCEPTION_H
