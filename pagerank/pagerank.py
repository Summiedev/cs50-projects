import os
import random
import re
import sys


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_distribution = dict()
    links = len(corpus[page])
    if corpus[page]:
        for p in corpus:
            prob_distribution[p] = (1 - damping_factor)/len(corpus)
        for p in corpus[page]:
            prob_distribution[p] += damping_factor/ links
    else:
        for p in corpus:
            prob_distribution[p]= 1/len(corpus)
    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank= dict().fromkeys(corpus.keys(),0)
    pages = [page for page in corpus]
    first_sample = random.choice(pages)

    for i in range(n):
        next=[]
        prob=[]
        dist = transition_model(corpus, first_sample, damping_factor)
        for key, value in dist.items():
            next.append(key)
            prob.append(value)
        first_sample = random.choices(next,weights=prob)[0]
        pagerank[first_sample]+=1
    for key in corpus:
        pagerank[key]/=n

    return pagerank



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    convergence = 0.001
    total_pages = len(corpus)
    page_rank = dict()
    random_choice_prob = (1 - damping_factor) / len(corpus)
    
    for page in corpus:
        page_rank[page] = 1/total_pages
    done = True
    while done:
        done = False
        newp = {key: value for key, value in page_rank.items()}
        for page in corpus:
            init =0
            for i in corpus:
                if page in corpus[i]:
                    init += page_rank[i] / len(corpus[i])
                if not corpus[i]:
                    init += 1 / len(corpus)
                    
            page_rank[page] = random_choice_prob + (damping_factor * init)       
            done = done or abs(page_rank[page]-newp[page]) > convergence

    su_m= sum(page_rank.values())
    page_rank = {key: value/su_m for key , value in page_rank.items()}

    return page_rank
   

if __name__ == "__main__":
    main()
