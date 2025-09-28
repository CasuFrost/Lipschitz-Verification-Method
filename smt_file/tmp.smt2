(set-logic QF_NRA)
(declare-fun x () Real)
(declare-fun y () Real)
(declare-fun z () Real)

(assert
    (^
      (+
            x
            
      ) 0 
    )
)

(assert
    (^
      (+
            z
            
      ) 0 
    )
)

(check-sat)
(get-model)
(exit)