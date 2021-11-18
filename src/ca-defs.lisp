#|
==============================================
-*- Mode: Lisp; Syntax: Common-Lisp -*-

File: ca-defs.lisp
Author: Vladimir Kulyukin
Description: Simple word definitions for CA
===============================================
|#

(in-package :user)

(define-ca-word 
    jack
    (concept nil (human :name (jack) :sex (male))))

(define-ca-word
    john
    (concept nil (human :name (john) :sex (male))))

(define-ca-word
    ann
    (concept nil (human :name (ann) :sex (female))))

(define-ca-word 
    ate
    (concept ?act (ingest :time (past)))
    (request (test (before ?act ?actor (animate)))
        (actions (modify ?act :actor ?actor)))
    (request (test (after ?act ?food (food)))
        (actions (modify ?act :object ?food))))

(define-ca-word
    bought
    (concept ?act (atrans :time (past)))
    (request (test (before ?act ?actor (animate)))
	    (actions (modify ?act :actor ?actor)))
    (request (test (after ?act ?obj (phys-obj)))
	    (actions (modify ?act :object ?obj))))

(define-ca-word 
    apple
    (concept nil (apple)))

(define-ca-word
    pear
    (concept nil (pear)))

(define-ca-word 
    an
    (mark ?x)
    (request (test (after ?x ?con (concept)))
	    (actions (modify ?con :ref (indef))))
    (request (test (after ?x ?con (concept)))
	    (actions (modify ?con :number (singular)))))

(define-ca-word
    a
    (mark ?x)
    (request (test (after ?x ?loc (location)))
        (actions (modify ?loc :ref (indef))))
    (request (test (after ?x ?loc (location)))
        (actions (modify ?loc :number (singular))))
    (request (test (after ?x ?obj (phys-obj)))
	    (actions (modify ?obj :ref (indef))))
    (request (test (after ?x ?obj (phys-obj)))
	    (actions (modify ?obj :number (singular)))))

;;; My defs:

(define-ca-word
    went
    (concept ?x (ptrans :time (past)))
    (request (test (after ?x ?loc (location)))
        (actions (modify ?x :to ?loc)))
    (request (test (before ?x ?actor (animate)))
	    (actions (modify ?x :actor ?actor)))
    (request (test (before ?x ?tran (atrans)))
	    (actions (modify ?x :from (to))))
    (request (test (before ?x ?obj (phys-obj)))
        (actions (modify ?x :object ?obj))))

(define-ca-word
    to
    (concept nil (to)))

(define-ca-word
    restaurant
    (concept nil (restaurant)))

(define-ca-word
    home
    (concept nil (home)))

(define-ca-word
    lobster
    (concept nil (lobster)))

;;; end-of-file

