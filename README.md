# transformer_exp
Implementing the transformer model (and doing some experiments on it), following Andrej Karpathy's lecture

# Model description
The transformer model was first introduced in the "Attention is All You Need" paper, written by researches from OpenAI. The transformer architecture is different from its predecessor, the Recurrent Neural Network (RNN) architecture mainly because of the great role it imparts to attention mechanisms. The transformer model is relatively simple to implement, fast to train due to parallelization possibilities and easy to inference from. 

# Training results 
The model was trained on the tiny_shakespeare data set, using my laptop's RTX 3060 graphics card. The lowest test loss achieved was about 1.79, with its hyperparameters being n_heads=6, n_embed=64, n_layer=6, batch_size=16, block_size=32...

Here is a snippet of text that was inferenced from the model:

***How like, which Whends in these bolizit:***
***And you, send never and would wed.***

***CLARWI***

***SLERD:***
***O knee, for his, awhith do you hatter? he nother***
***Is wall names! where, but one***
***Who man for, and, tito when as thenter I shall propech***
***To sustion of the with awactius slong recutedtion***
***Batten ven Listongelity to me?***

***First KING MARGARET:***
***Ay, I am***
***Mestres your nor flame, who you I have of finind,***
***As when liss friend we'll***
***fign's handing darencect south follns I yours***
***Thou so maked: But byt ware' yad.***
***It will a now anly wellong.***

***HASTINGS:***
***What's or patessiel, there meake o'***
***the eather brefling your to know?***

***Brother:***
***And, evinge with; foul a longed you the will the warticy now for the and know with but I traice frent a let you; for with make wear to to intentant***
***To rue delire the pril to way?***

***STHAY CARD IV:***
***Wat that you htad not her but***
***would peeceing untis meightens. What, blinted***
***Hinghave and but when your little dotuccagess***
***Of stakes the gonesing, lady music one it.***

***BUCHARD III:***
***Mestand me pealling.***

***RICHARD IIII:***
***Hark I melt Menteman, he entends.***

***PETHLORazeNeon:***
***Let's see my ot***
***the; day, wherest Boninnes too head.***

***LORD ATHARY Call'd.***

***LUCIO:***
***I hollmned condure!***
***The lone; who***
***art honour, wonlds, my-rassone.***

***PELIZfly:***
***Twit Jout pities I wall.***

***ANGELO:***
***Well, Isake my briant.***

***LEOurs, Bein'd longgerablief, for rowher***
***the wise; for one contun one, that.***

***DUGHY:***
***Malike but somes wouldeders.***

***GLOUCESTER:***
***That woeld but you, do long.***

***Pircange:***
***Madamm can onthy to rather***
***But this, much anter, dateful he how!***

***YORK:***
***Thousant then too resolove Ond, good straits, trobe, blay, pitces both Mongin with your is the are fednow?***

***BUCQOLICESTHARD III:***
***Whose look. Sir! on tispraint; is callooks make***
***you and the scann'd and toundspen.***

***BAPTULY:***
***O, my son, Thrishoughtincest not like.***

***QUEEN,***
***In your weak you, then you and conemtantes,***
***We we tan have from from one todge out witwer'n.***

***HORWIS RICHARD IIII:***
***He have like it his hest, mambuch desdired with comes and so by,***
***Constent me he switt***
