from string import punctuation


class PQBase:
    """
    [PythonLabs] A priority queue is an ordered sequence, similar to a list, except items
    can only be added by calling the insert method, which adds a single item so the queue
    remains sorted, and can only be removed by calling pop, which removes a single item 
    from the front.  This class defines the base functionality of a priority queue; labs
    such as BitLab and ElizaLab will define their own PriorityQueue classes as extensions
    of PQBase.
    """
    def __init__(self):
        self._q = []
        
    def __repr__(self):
        return "[" + ", ".join(map(str,self._q)) + "]"
        
    def __len__(self):
        "[PythonLabs] Return the number of items in this queue"
        return len(self._q)

    def insert(self, x):
        "[PythonLabs] Add an item to this queue, keeping the queue sorted"
        i = 0
        while i < len(self._q):
            if x < self._q[i]: break
            i += 1
        self._q.insert(i, x)
        return self
    
    def pop(self):
        "[PythonLabs] Remove the first item from this queue"
        if len(self._q) > 0:
            return self._q.pop(0)
        else:
            return None
            
    def clear(self):
        " [PythonLabs] Delete all items in this queue."
        self._q = []
            
    def __getitem__(self, x):
        "[PythonLabs] Access individual items in this queue"
        return self._q[x]
        
    def __iter__(self):
        "[PythonLabs] Iterate over all items in this queue"
        for x in self._q:
            yield x
    

class WordQueue(PQBase):
    """
    [SpamLab] A WordQueue is an ordered collection of "interesting" words.  Add words by calling
    insert.  Words are automatically removed when the queue grows beyond its maximum size
    (specified when the queue is created).
    """
    def __init__(self, size):
        "[SpamLab] Create a new word queue, initially empty."
        super().__init__()
        self._capacity = size
        self._on_canvas = False
        
    def insert(self, word, prob):
        "[SpamLab] Save a word and its probabilty in the queue"
        score = abs(prob - 0.5)
        i = 0
        while i < len(self._q) and score < self._q[i][2]:
            i += 1
        if i < len(self._q) and word == self._q[i][0]:
            return;
        if i < self._capacity:
            self._q.insert(i, (word, prob, score))
        last = None
        if len(self._q) > self._capacity:
            last = self._q.pop(self._capacity)
        if self._on_canvas:
            update_view(self, i, word, prob, score, last)

    def words(self):
        "[SpamLabs] Generate the sequence of words in the queue"
        for x in self._q:
            yield x[0]

    def probs(self):
        "[SpamLabs] Generate the sequence of probabilities of words in the queue"
        for x in self._q:
            yield x[1]

def view_queue(wq, **user_options):
    options = dict(_queue_view_options)
    options.update(user_options)
    cw = (3 * _qx) + (2 * _box_width) + 20
    ch = (2 * _qy) + (wq._capacity * _box_height) + 20
    Canvas.init(cw, ch, "SpamLab: Word Queue")
    Canvas.delay = 0.5
    view = QueueView(wq, options)
    Canvas.register(view)
    wq._on_canvas = True
    y = _qy
    for i in range(0,wq._capacity):
        view.boxes.append(Canvas.Rectangle(_qx, y, _qx + _box_width, y + _box_height, fill = options['box_bg'], outline = options['box_line'], tag = "box%s" % i))
        y += _box_height
    return view
    

def spamcity(w, pbad, pgood):
    #Compute the probability a messaage is spam when it contains a word w.
    #The dictionaries pbad and pgood hold p(w|spam) and p(w|good), respectively.
    if w in pbad and w in pgood:
        return pbad[w] / ( pbad[w] + pgood[w] )
    else:
        return None

def tokenize(s):
    "Return the list of words in string s"
    a = [ ]
    for x in s.split():
        a.append( x.strip(punctuation).lower() )
    return a
    
def wf(fn):
    "Make a dictionary of word frequencies"
    count = { }
    for line in open(fn):
        for w in tokenize(line):
            count.setdefault(w, 0)
            count[w] += 1
    return count
    
def load_probabilities(fn):
    prob = { }
    with open(fn, 'r') as f:
        for line in f:
            p, w = line.split()
            prob.update({w: (float(p))})
    return prob

def combined_probability(queue):
    #queue is of class WordQueue defined above
    p = q = 1.0
    for x in queue.probs():
        p *= x
        q *= (1.0 - x)
    return p / (p + q)

def pspam(fn):
    "Compute the probability the message in file fn is spam"
    queue = WordQueue(15) #queue of interesting words
    pgood = load_probabilities('good.txt')
    pbad = load_probabilities('bad.txt')
    for line in open(fn):
        for word in tokenize(line):#word is lower case without punctuation
            p = spamcity(word, pbad, pgood) #Prob{spam | word}
            if p != None:
                queue.insert(word, p) #add the word to the queue
    return combined_probability(queue)
