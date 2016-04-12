Agents
==================

The NeuralAgent class wraps up all methods for exploring, training and testing in any given environment.

It relies on controllers to modify its hyper parameters through time and to decide how the validation/test procedures should happen. Controllers are attached to an agent using the agent.attach(Controller) method. 

All controllers should inherit from the base Controller class (which does nothing when receiving the various signals emitted by an agent). The following methods are defined in this base controller class:

* `__init__(self)`: Activate the controller.
All controllers inheriting this class should call this method in their own `__init()__` using `super(self.__class__, self).__init__()`.
* `setActive(self, active)`: Activate or deactivate this controller. A controller should not react to any signal it receives as long as it is deactivated. For instance, if a controller maintains a counter on how many episodes it has seen, this counter should not be updated when this controller is disabled.
* `OnStart(self, agent)`: Called when the agent is going to start working (before anything else). This corresponds to the moment where the agent's `run()` method is called.
* `OnEpisodeEnd(self, agent, terminalReached, reward)`: Called whenever the agent ends an episode, just after this episode ended and before any `OnEpochEnd()` signal could be sent.

Learning rate controller
------------

A controller that modifies the learning rate periodically upon epochs end.

### Parameters

* initialLearningRate [float] - The learning rate upon agent start
* learningRateDecay [float] - The factor by which the previous learning rate is multiplied every
                [periodicity] epochs.
* periodicity [int] - How many epochs are necessary before an update of the learning rate occurs

Epsilon Controller
------------

A controller that modifies the probability "epsilon" of taking a random action periodically.

### Parameters

* initialE [float] - Start epsilon
* eDecays [int] - How many updates are necessary for epsilon to reach eMin
* eMin [float] - End epsilon
* evaluateOn [str] - After what type of event epsilon shoud be updated periodically. Possible values: 'action', 'episode', 'epoch'.
* periodicity [int] - How many [evaluateOn] are necessary before an update of epsilon occurs
* resetEvery [str] - After what type of event epsilon should be reset to its initial value. Possible values: 'none', 'episode', 'epoch'.

Discount Factor Controller
------------

A controller that modifies the qnetwork discount periodically.

### Parameters

* initialDiscountFactor [float] - Start discount
* discountFactorGrowth [float] - The factor by which the previous discount is multiplied every [periodicity] epochs.
* discountFactorMax [float] - Maximum reachable discount
* periodicity [int] - How many epochs are necessary before an update of the discount occurs

Interleaved Test Epoch Controller
------------

A controller that interleaves a test epoch between training epochs of the agent.

### Parameters

* id [int] - The identifier (>= 0) of the mode each test epoch triggered by this controller will belong to. Can be used to discriminate between datasets in your Environment subclass (this is the argument that will be given to your environment's reset() method when starting the test epoch).
* epochLength [float] - The total number of transitions that will occur during a test epoch. This means that this epoch could feature several episodes if a terminal transition is reached before this budget is  exhausted.
* controllersToDisable [list of int] - A list of controllers to disable when this controller wants to start a test epoch. These same controllers will be reactivated after this controller has finished dealing with its test epoch.
* periodicity [int] - How many epochs are necessary before a test epoch is ran (these controller's epochs included: "1 test epoch on [periodicity] epochs"). Minimum value: 2.
* showScore [bool] - Whether to print an informative message on stdout at the end of each test epoch, about  the total reward obtained in the course of the test epoch.
* summarizeEvery [int] - How many of this controller's test epochs are necessary before the attached agent's  summarizeTestPerformance() method is called. Give a value <= 0 for "never". If > 0, the first call will occur just after the first test epoch.

Trainer Controller
------------

A controller that makes the agent train on its current database periodically.

### Parameters

* evaluateOn [str] - After what type of event the agent shoud be trained periodically. Possible values: 'action', 'episode', 'epoch'. The first training will occur after the first occurence of [evaluateOn].
* periodicity [int] - How many [evaluateOn] are necessary before a training occurs
* showAvgBellmanResidual [bool] - Whether to show an informative message after each episode end (and after a  training if [evaluateOn] is 'episode') about the average bellman residual of this episode
* showEpisodeAvgVValue [bool] - Whether to show an informative message after each episode end (and after a training if [evaluateOn] is 'episode') about the average V value of this episode

Verbose Controller
------------

A controller that prints various agent information periodically:
* Count of passed [evaluateOn]
* Agent current learning rate
* Agent current discount factor
* Agent current epsilon

### Parameters

* evaluateOn [str] - After what type of event the printing should occur periodically. Possible values: 'action', 'episode', 'epoch'. The first printing will occur after the first occurence of [evaluateOn].
* periodicity [int] - How many [evaluateOn] are necessary before a printing occurs

Find Best Controller
------------

A controller that finds the neural net performing at best in validation mode (i.e. for mode = [validationID]) 
and computes the associated generalization score in test mode (i.e. for mode = [testID], and this only if [testID] 
is different from None). This controller should never be disabled by InterleavedTestControllers as it is meant to 
work in conjunction with them.

At each epoch end where this controller is active, it will look at the current mode the agent is in. 

If the mode matches [validationID], it will take the total reward of the agent on this epoch and compare it to its 
current best score. If it is better, it will ask the agent to dump its current nnet on disk and update its current 
best score. In all cases, it saves the validation score obtained in a vector.

If the mode matches [testID], it saves the test (= generalization) score in another vector. Note that if [testID] 
is None, no test mode score are ever recorded.

At the end of the experiment (OnEnd), if active, this controller will print information about the epoch at which 
the best neural net was found together with its generalization score, this last information shown only if [testID] 
is different from None. Finally it will dump a dictionnary containing the data of the plots ({n: number of 
epochs elapsed, ts: test scores, vs: validation scores}). Note that if [testID] is None, the value dumped for the
'ts' key is [].

### Parameters

* validationID [int] - See synopsis
* testID [int] - See synopsis
* unique_fname [str] - A unique filename (basename for score and network dumps).