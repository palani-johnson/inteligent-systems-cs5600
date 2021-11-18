#|
===========================================================
-*- Mode: Lisp; Syntax: Common-Lisp -*-

File: ca-loader.lisp
Author: Vladimir Kulyukin

Bugs to vladimir kulyukin in canvas
===========================================================
|#

(in-package :user)

;;; change this to the directory where the ca files are.
(defparameter *ca-dir* "/home/palani/Projects/School/inteligent-systems-cs5600/src/")

(defparameter *files* '("ca-utilities" "cd" "ca" "ca-functions" "ca-lexicon"))

(defun ca-loader (files &key (mode :lisp))
  (dolist (a-file files t)
    (load (concatenate 'string
            *ca-dir*
            a-file
            (ecase mode
              (:lisp ".lisp")
              (:cl  ".cl")
              (:fasl ".fasl"))))))
