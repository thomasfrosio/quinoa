#include <noa/common/Exception.h>
#include <noa/String.h>

#include "quinoa/Exception.h"

std::string qn::Exception::backtrace() {
    const std::vector<std::string> backtrace_vector = ::noa::Exception::backtrace();

    std::string backtrace_message;
    for (size_t i = 0; i < backtrace_vector.size(); ++i)
        backtrace_message += noa::string::format("[{}]: {}\n", i, backtrace_vector[i]);
    return backtrace_message;
}
