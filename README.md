This is a framework for working with Binary Quadratic Problems.

Class bqp is about reading instances from file, generating random instances, converting boolean instances to spin and so on.
All methods below will work with the spin representation.

Class SDP takes a bqp instance as an input and solves it with prescribed parameters. User may specify rank constraint, rounding procedure and number of rounds, precision for optimality criteria and etc. Regularised SDP will soon be added as an option as well as other rounding procedures.
