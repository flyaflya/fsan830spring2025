# The Most Ambitious Class on The Internet - FSAN830 Business Process Innovation

This repository documents the 'Innovation Track' of UD's Spring 2025 FSAN830 Business Process Innovation class. Our ambitious goal is to revolutionize horse racing predictions by implementing a state-of-the-art statistical model (BART) within an automated workflow, testing academic theory against real-world market outcomes. Through this public repository, students gain hands-on experience with modern DevOps practices, agile methodologies, and the power of having 'skin in the game' – skills essential for driving real-world innovation. By combining rigorous statistical analysis with industry-standard development practices, we're preparing students to make meaningful impacts in their future careers.

# Our First Mission - Class Pages

Our first mission is to start populating this class website that will serve as a central hub for our class.  We will use the internet to its fullest extent to accomplish this goal.

> Provide some details about yourself on your profile page and add a picture.  Ensure it includes some notion of your aspirations in using data science and mention any passions that might overlap with the course material.

# Our First Mission - Class Pages

Our first mission is to start populating this class website that will serve as a central hub for our class. We will use the internet to its fullest extent to accomplish this goal.

> Provide some details about yourself on your profile page and add a picture (50x50px). Ensure it includes some notion of your aspirations in using data science and mention any passions that might overlap with the course material.

### Instructor
| <img src="images/fleischhacker_300x300.png" width="100" height="100"> |
|:---:|
| [Fleischhacker's Profile](Fleischhacker.md) |

### Student Profiles
| <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> |
|:---:|:---:|:---:|
| [Aghababaei's Profile](Aghababaei.md) | [Buskin's Profile](Buskin.md) | [Dong's Profile](Dong.md) |
| <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> |
| [G. Chen's Profile](GChen.md) | [Guo's Profile](Guo.md) | [He's Profile](He.md) |
| <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> |
| [Li's Profile](Li.md) | [Neshastehriz's Profile](Neshastehriz.md) | [Okediran's Profile](Okediran.md) |
| <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> | <img src="images/race_horse_avatar_300x300.png" width="100" height="100"> |
| [S. Chen's Profile](SChen.md) | [Wagner's Profile](Wagner.md) | [Zou's Profile](Zou.md) |


# Motivational Video

This video is perhaps the most important background video for the second half of our semester together.

[Link to Video](https://youtu.be/jsBpNCxxlNE?si=4LHByThKyIkbBJx1&t=5500)

<iframe width="560" height="315" src="https://www.youtube.com/embed/jsBpNCxxlNE?si=QiqzbgQRvpnYT1B1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

We will watch the above video in segments throughout the semester.

# Important Papers

* Hugh A. Chipman. Edward I. George. Robert E. McCulloch. *BART: Bayesian additive regression trees.* Ann. Appl. Stat. 4 (1) 266 - 298, March 2010.
* Hill, J. L. (2011). *Bayesian nonparametric modeling for causal inference.* Journal of Computational and Graphical Statistics, 20(1), 217-240.
* Hill, J., Linero, A., & Murray, J. (2020). *Bayesian additive regression trees: A review and look forward. Annual Review of Statistics and Its Application*, 7(1), 251-278.


# Beliefs

Observational Data:
* it is nearly impossible to correctly leverage statistical methods on **observational** data to make reliable predictions about **causal** effects, e.g. confounding, selection bias, garbage can regressions, etc.
* the only way to reliably make causal predictions is to leverage **interventional** data from properly conducted randomized experiments.
* observational data is not useless, but it is important to always be aware of the limitations of observational data and validate any causal predictions with interventional data.
* predictive models, while not perfect, are still useful for making predictions about future outcomes.

Feedback Loops:
* Academic institutions select for ideas that are novel and intellectually appealing rather than proven and resilient, creating an ecosystem where being impressively wrong is often more rewarded than being boringly correct.
* When scholars aren't required to place personal capital or reputation at risk, peer review becomes an exercise in academic aesthetics rather than a meaningful test of an idea's survival value.
* The filtering mechanisms of peer review and real-world implementation are fundamentally different - one selects for theoretical elegance and methodological rigor, while the other ruthlessly eliminates ideas that fail to deliver tangible results.

Goodhart's Law - when a measure becomes a target, it ceases to be a good measure:
* Scholars optimize for what's measurable (citations, h-index, publication count) rather than what's valuable (reproducibility, real-world impact, practical utility), creating an avalanche of technically sophisticated but fundamentally hollow research.
* The academic incentive system has become a perfect example of the very phenomenon it should study - the complete divergence of proxy measures from the underlying quality they were meant to track.

Horse racing will be our laboratory:
* Racing markets represent a rare interventional laboratory where probabilistic predictions face immediate financial consequences - models that fail to capture real relationships are rapidly punished through capital destruction, unlike in traditional academic settings.
* Horse racing data provides a perfect testing ground because the environment combines genuine uncertainty, rapid feedback cycles, and most importantly, skin in the game through betting markets that force probabilistic estimates to survive contact with reality.
* The use of horse racing data cleverly sidesteps the traditional academic trap of optimizing for mathematical elegance over practical validity - a model's success or failure is determined not by peer review, but by its ability to generate predictions that survive in a competitive market environment.

Bayesian Additive Regression Trees (BART):
* BART is a non-parametric regression model that uses Bayesian inference to estimate the relationship between a dependent variable and one or more independent variables.
* BART is, to me, the most promising new development in statistical modeling, it has performed extremely well in causal inference competitions (google Atlantic Causal Inference Competition).

Our innovation:
* We will use BART to make predictions about horse racing outcomes.
* We will then use the predictions to make bets on horse racing outcomes.
* We will then evaluate the performance of the model.
* Through real-world application, we will discover and patch many holes in the academic literature on BART while validating its practical utility.

Why horse racing and not say, the stock market?

* Faster Feedback Cycles
  * Horse races provide immediate feedback - typically within minutes. Stock market strategies often require months or years to validate due to noise and the need for sufficient sample size. This rapid feedback allows for faster iteration and learning.
* Horse racing has fewer confounding variables compared to stocks. 
  * While still complex, the key variables (track conditions, horse history, jockey, etc.) are more bounded and measurable. Stock prices are affected by countless global factors, making it harder to isolate the effectiveness of your modeling approach.
* Market Inefficiencies
  * The horse racing market tends to be less efficient than major stock markets. Large institutional investors and algorithmic traders have largely eliminated easily exploitable patterns in stocks. Horse racing still has more retail participants and potential inefficiencies that a good model could identify.
* Data Quality and Accessibility
  * Horse racing data tends to be more standardized and complete within its domain. While stock market data is abundant, the truly valuable predictive signals are often private or expensive. The playing field is more level in horse racing data.
* Lower Capital Requirements
  * You can test betting strategies with relatively small amounts of capital and get statistically significant results. Testing stock market strategies often requires substantial capital to overcome transaction costs and achieve meaningful sample sizes.
* Independence Between Events
  * Each horse race is largely independent of other races. Stock market moves are highly correlated across securities and time periods, making it harder to build a robust sample size of truly independent observations.




















