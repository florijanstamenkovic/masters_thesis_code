"""
Scans the logs of ngram model evaluations
and shows only the resutls.
"""
import os
import logging

log = logging.getLogger(__name__)


def main():

    logging.basicConfig(level=logging.INFO, format="")
    log.info("Ngram model evaluation")

    for dir in os.listdir("."):
        path = os.path.join(dir, "eval.log")
        if not os.path.exists(path):
            continue

        log.info("\n%s", dir)
        with open(path) as f:
            lines = [l.strip() for l in f.readlines()]
            models = []
            scores = []
            for l in lines:
                if l.startswith("Evaluating model:"):
                    models.append(l[18:])
                if l.startswith("Score:"):
                    scores.append(l[7:])

            for s, m in zip(scores, models):
                log.info("\t%s - %s", s, m)


if __name__ == '__main__':
    main()
