#pragma once

#include <experimental/source_location>
#include <filesystem>

namespace hpm {

using SourceLoc = std::experimental::source_location;

auto getPath(std::string const &newFile,
             std::string const &thisFile = SourceLoc::current().file_name())
    -> std::string {
  return std::filesystem::path(thisFile).replace_filename(newFile);
}

} // namespace hpm
