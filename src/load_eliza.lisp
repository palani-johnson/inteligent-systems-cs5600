;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-

#|
=========================================================
Module: load-eliza.lisp: 
Description: a load for three eliza modules.
bugs to vladimir kulyukin in canvas
=========================================================
|#

;;; change this parameter as needed.
(defparameter *eliza-path* 
  "/home/palani/Projects/School/inteligent-systems-cs5600/src/")

;;; the files that comprise ELIZA.
(defparameter *eliza-files* 
  '("auxfuns.lsp" "eliza.lsp")) 

(defun load-eliza-aux (path files)
  (mapc #'(lambda (file)
	    (load (concatenate 'string path file)))
	files))

;;; load ELIZA
(defun load-eliza ()
  (load-eliza-aux *eliza-path* *eliza-files*))

	
