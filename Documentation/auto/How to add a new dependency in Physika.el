(TeX-add-style-hook "How to add a new dependency in Physika"
 (lambda ()
    (TeX-add-symbols
     '("reals" 1)
     '("mb" 1)
     '("limit" 2)
     '("jump" 1)
     '("subheading" 1)
     "grad"
     "urlfont")
    (TeX-run-style-hooks
     "geometry"
     "graphicx"
     "graphics"
     "algorithmic"
     "amssymb"
     "amsmath"
     "multicol"
     "latex2e"
     "art11"
     "article"
     "11pt"
     "fullpage")))

