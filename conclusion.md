


# 1
## a
### 0,8 looks a good threshold. High treshholds implies in a model that no recognize a person with a sweater with similar color of the floor (dont capture small changes on pixels), and a very low threshold will capture a lot of noise.

## b
### As we expect, there is a problem when de background model contains an object that will move, this creates a ghost effect with the position of the object in the background and the new position after move.

## c
### Small delays difficulte capturing slow objects, since they changed a bit. Big delays captures a lot of ghosts for fast objects. We should use small delays for fast objects and big delays for slow objects.

## d
### So this methods allow us to see the movement trace from an object, the postions before t - d will have less intensity, and the trace between t and t - d will have less intensity, but the position t and t - d will have value 1 (max intensity). The disvantage is that now we have the history of positions of an object but not exactly yhe object position.


# 2
## a 
### For small T the model capture a lot of noise like a granulate pattern on the screan, glass reflection with false positive. The model cant adapt to changes on lightining, brightness and noises. For big T we loss information about the foreground since changes start to not being captured.

## b
