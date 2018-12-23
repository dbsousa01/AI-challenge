# AI coding challenge

## Description

The first thing you need to be able to do is to identify these languages properly.

We need you to build a service that provided with a text, identifies the language in which it is 
written, and provides that answer.
We provide you with the initial repo to implement this, and some requirements that the service 
should satisfy. 

Whether you choose to implement an existing approach or compile one, make sure you document it
and explain your reasoning.

### Tasks

1 - Implement a Language Identification service that returns the language code of the language in which the text is written. The provided data and test will
target Spanish (ES), Portuguese (PT-PT) and English (EN)

2 - Train the system to distinguish between language variants. In this case we wish to distinguish between European Portuguese (PT-PT) and Brazilian Portuguese (PT-BR)

3(not completed) - Implement a deep learning model (recommended: a BILSTM tagger) to detect code switching (language mixture) and return both a list of tokens and a list with one language label per token.
To simplify we are going to focus on English and Spanish, so you only need to return for each token either 'en', 'es' or 'other'

*See more information about tasks 1 and 2 in langid folder, and about task3 in code_switching folder*
