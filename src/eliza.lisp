;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-

#|
=========================================================
Module: eliza.lisp: 
Description: A version of ELIZA that takes inputs without 
paretheses around them unlike eliza1.lisp.
Bugs to vladimir kulyukin in canvas
=========================================================
|#

;;; ==============================

(defun rule-pattern (rule) (first rule))
(defun rule-responses (rule) (rest rule))

(defun read-line-no-punct ()
  "Read an input line, ignoring punctuation."
  (read-from-string
    (concatenate 'string "(" (substitute-if #\space #'punctuation-p
                                            (read-line))
                 ")")))

(defun punctuation-p (char) (find char ".,;:`!?#-()\\\""))

;;; ==============================

(defun use-eliza-rules (input)
  "Find some rule with which to transform the input."
  (some #'(lambda (rule)
            (let ((result (pat-match (rule-pattern rule) input)))
              (if (not (eq result fail))
                  (sublis (switch-viewpoint result)
                          (random-elt (rule-responses rule))))))
        *eliza-rules*))

(defun switch-viewpoint (words)
  "Change I to you and vice versa, and so on."
  (sublis '((i . you) (you . i) (me . you) (am . are) (my . your) (your . my))
          words))

(defparameter *good-byes* '((good bye) (see you) (see you later) (so long)))

(defun eliza ()
  "Respond to user input using pattern matching rules."
  (loop
    (print 'eliza>)
    (let* ((input (read-line-no-punct))
           (response (flatten (use-eliza-rules input))))
      (print-with-spaces response)
      (if (member response *good-byes* :test #'equal)
	  (RETURN))))
  (values))

(defun print-with-spaces (list)
  (mapc #'(lambda (x) (prin1 x) (princ " ")) list))

(defun print-with-spaces (list)
  (format t "~{~a ~}" list))

;;; ==============================

(defparameter *eliza-rules*
  '(
    ;;; rule 1
    (((?* ?x) hello (?* ?y))      
    (How do you do.))

    ;;; rule 2
    (((?* ?x) computer (?* ?y))
     (Do computers worry you?)
     (What do you think about machines?)
     (Why do you mention computers?)
     (What do you think machines have to do with your problem?))

    ;;; rule 3
    (((?* ?x) name (?* ?y))
     (I am not interested in names))

    ;;; rule 4
    (((?* ?x) sorry (?* ?y))
     (Please don't apologize)
     (Apologies are not necessary)
     (What feelings do you have when you apologize))

    ;;; rule 5
    (((?* ?x) remember (?* ?y)) 
     (Do you often think of ?y)
     (Does thinking of ?y bring anything else to mind?)
     (What else do you remember)
     (Why do you recall ?y right now?)
     (What in the present situation reminds you of ?y)
     (What is the connection between me and ?y))

    ;;; rule 6
    (((?* x) good bye (?* y))
     (good bye))

    ;;; rule 7
    (((?* x) so long (?* y))
     (good bye)
     (bye)
     (see you)
     (see you later))

    ;;; ========== your rules begin
    (((?* ?x) you (?* ?y) fortunes (?* ?z))
     (i dabble)
     (the universe sent you here for a reason))
    
    (((?* ?x) fortune telling (?* ?y))
     (i sensed you would say that))

    (((?* ?x) my fortune (?* ?y))
     (What do you want to know about?)
     (I can only see what wants to be known))

    (((?* ?x) when (?* ?y) die (?* ?z))
     (ummm you dont want to know that)
     (i sense that an answer will not ease your troubles)
     (someday in the future))

    (((?* ?x) will i (?* ?y) love (?* ?z))
     (nah fam not a chance)
     (love will find you when you are ready))

    (((?* ?x) does (?* ?y) love me (?* ?z))
     (what is this? highschool?)
     (?y dosent even know you exist))

    (((?* ?x) where are (?* ?y))
     (wherever you left them))

    (((?* ?x) dont believe (?* ?y))
     (well thats just what the magic ball told me)
     (hey dont shoot the messanger))

    (((?* ?x) will i (?* ?y) rich (?* ?z))
     (not likely)
     (not with that attitude)
     (the system is rigged against you)
     (rich with money? no certainly not)
     (no)
     (nope)
     (never)
     (haha thats rich)
     (not in a million years)
     (if you work hard and grind for 40 years you will still be broke)
     (you wont be giving bezos a run for his money but you will be wealthy))
    
    (((?* ?x) will i (?* ?y) money (?* ?z))
     (not likely)
     (not with that attitude)
     (the system is rigged against you)
     (rich with money? no certainly not)
     (no)
     (nope)
     (never)
     (haha thats rich)
     (not in a million years)
     (if you work hard and grind for 40 years you will still be broke)
     (you wont be giving bezos a run for his money but you will be wealthy))

    (((?* ?x) should i (?* ?z))
     (?z if you wish but it will make you no happier)
     (?z ? just dont))

    (((?* ?x) how (?* ?y) i die (?* ?z))
     (you will be hit by a bus while wearing a red hat)
     (you will be eaten by a tiger while in minnesota)
     (an elephant will step on you during a parade)
     (zapped by a lightning bolt while standing in the parcking lot of a costco)
     (medical malpractice in a plane)
     (blown up in a horiffic leaf blower incident))

    (((?* ?x) I dont (?* ?y) fortune tellers (?* ?z))
     (well i dont ?y you))

    (((?* ?x) does (?* ?y) exist (?* ?z))
     (does anything really exist?)
     (?y exists in the hearts and minds of us all)
     (i dont think im qualified to say))

    (((?* ?x) do (?* ?y) exist (?* ?z))
     (does anything really exist?)
     (?y exist in the hearts and minds of us all)
     (ask a physicist))

    (((?* ?x) my purpose (?* ?z))
     (to pass butter)
     (to stick it to the man)
     (to be a cog in the machine))

    (((?* ?x) find (?* ?y) job (?* ?z))
     (you just gotta fill out 400 more applications)
     (i heard walmart is hiring))
    
    (((?* ?x) can you (?* ?y) future (?* ?z))
     (we live in a deterministic universe)
     (i can see everything)
     (omniscience is a blessing and a curse))

    (((?* ?x) what (?* ?y) years (?* ?z))
     (expect the unexpected)
     (some crazy stuff)
     (ehh its kinda boring looking))
    
    (((?* ?x) why is (?* ?y) future so (?* ?z))
     (thats not how i saw it)
     (?z is just an opinion))

    ;;; ========== your rules end

   ))

;;; ==============================
#|
ELIZA> hey do you do fortunes
THE UNIVERSE SENT YOU HERE FOR A REASON 
ELIZA> well what is my fortune
WHAT DO YOU WANT TO KNOW ABOUT? 
ELIZA> umm will i ever find love
LOVE WILL FIND YOU WHEN YOU ARE READY 
ELIZA> when will i die
UMMM YOU DONT WANT TO KNOW THAT 
ELIZA> should i invest in stocks
INVEST IN STOCKS IF YOU WISH BUT IT WILL MAKE YOU NO HAPPIER 
ELIZA> do ghosts exist
DOES ANYTHING REALLY EXIST? 
ELIZA> does god exist
GOD EXISTS IN THE HEARTS AND MINDS OF US ALL 
ELIZA> can you really see the future
I CAN SEE EVERYTHING 
ELIZA> how will i die
BLOWN UP IN A HORIFFIC LEAF BLOWER INCIDENT 
ELIZA> i dont believe that one bit
HEY DONT SHOOT THE MESSANGER 
ELIZA> will i ever be rich
HAHA THATS RICH
ELIZA> does that girl next door love me
WHAT IS THIS? HIGHSCHOOL? 
ELIZA> how long will it take me to find a job
YOU JUST GOTTA FILL OUT 400 MORE APPLICATIONS 
ELIZA> what does the next five years hold for me
EXPECT THE UNEXPECTED 
ELIZA> oh im done here so long
BYE
|#
