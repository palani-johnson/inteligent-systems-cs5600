

(load "ca-loader.lisp")

(ca-loader *files*)

(load "ca-defs.lisp")

(load "sam.lisp")

(setf *restaurant-story*
    '((jack went to a restaurant)
      (jack ate a lobster)
      (jack went home)))

(setf *shopping-story*
    '((ann went to a store)
      (ann bought a kite)
      (ann went home)))


;;; Next
(sam (sents-to-cds *restaurant-story*))

(sam (sents-to-cds *shopping-story*))