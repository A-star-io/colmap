// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_RETRIEVAL_INVERTED_FILE_ENTRY_H_
#define COLMAP_SRC_RETRIEVAL_INVERTED_FILE_ENTRY_H_

#include <bitset>
#include <fstream>

namespace colmap {
namespace retrieval {

// Models an inverted file entry. The template defines the dimensionality of
// the binary string used to approximate the descriptor.
// This class is based on an original implementation by Torsten Sattler.
template <int N>
struct InvertedFileEntry {
  void Read(std::ifstream* ifs);
  void Write(std::ofstream* ofs) const;

  // The identifier of the image this entry is associated with.
  int image_id = -1;

  // The binary signature in the Hamming embedding.
  std::bitset<N> descriptor;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int N>
void InvertedFileEntry<N>::Read(std::ifstream* ifs) {
  static_assert(N <= 64, "Dimensionality too large");
  static_assert(sizeof(unsigned long) >= 8,
                "Expected unsigned long to be at least 8 byte");

  int32_t image_id = 0;
  ifs->read(reinterpret_cast<char*>(&image_id), sizeof(int32_t));
  image_id = static_cast<int>(image_id);

  uint64_t bin_desc = 0;
  ifs->read(reinterpret_cast<char*>(&bin_desc), sizeof(uint64_t));
  descriptor = std::bitset<N>(bin_desc);
}

template <int N>
void InvertedFileEntry<N>::Write(std::ofstream* ofs) const {
  static_assert(N <= 64, "Dimensionality too large");
  static_assert(sizeof(unsigned long) >= 8,
                "Expected unsigned long to be at least 8 byte");

  ofs->write(reinterpret_cast<const char*>(&image_id), sizeof(int32_t));

  uint64_t bin_desc = static_cast<uint64_t>(descriptor.to_ulong());
  ofs->write(reinterpret_cast<const char*>(&bin_desc), sizeof(uint64_t));
}

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_INVERTED_FILE_ENTRY_H_
