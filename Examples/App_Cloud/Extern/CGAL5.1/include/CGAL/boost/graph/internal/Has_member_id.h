// Copyright (c) 2016 GeometryFactory (France).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/BGL/include/CGAL/boost/graph/internal/Has_member_id.h $
// $Id: Has_member_id.h 52164b1 2019-10-19T15:34:59+02:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s) : Jane Tournois


#ifndef CGAL_HAS_MEMBER_ID_H
#define CGAL_HAS_MEMBER_ID_H

namespace CGAL {
namespace internal {

  template <typename Type>
  class Has_member_id
  {
    typedef char yes[1];
    typedef char no[2];

    struct BaseWithId
    {
      void id(){}
    };
    struct Base : public Type, public BaseWithId {};

    template <typename T, T t>
    class Helper{};

    template <typename U>
    static no &check(U*, Helper<void (BaseWithId::*)(), &U::id>* = 0);

    static yes &check(...);

  public:
    static const bool value = (sizeof(yes) == sizeof(check((Base*)(0))));
  };

}  // internal
}  // cgal

#endif /* CGAL_HAS_MEMBER_ID_H */
