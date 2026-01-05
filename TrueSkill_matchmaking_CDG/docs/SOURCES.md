# Sources et Bibliographie

## 📚 Publications Académiques

### TrueSkill - Papers Fondateurs

1. **Herbrich, R., Minka, T., & Graepel, T. (2006)**  
   *TrueSkill™: A Bayesian Skill Rating System*  
   In Advances in Neural Information Processing Systems 19 (NIPS 2006)  
   **Lien** : [PDF](https://papers.nips.cc/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf)  
   **Résumé** : Paper fondateur qui introduit TrueSkill, décrit l'algorithme d'inférence (Expectation Propagation) et présente les résultats initiaux.

2. **Minka, T.  (2013)**  
   *TrueSkill 2: An improved Bayesian skill rating system*  
   Microsoft Research Technical Report  
   **Lien** :  [Microsoft Research](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)  
   **Résumé** : Version améliorée avec support des matchs en équipes hétérogènes et dynamique temporelle.

3. **Dangauthier, P., Herbrich, R., Minka, T., & Graepel, T. (2007)**  
   *TrueSkill Through Time: Revisiting the History of Chess*  
   Advances in Neural Information Processing Systems 20 (NIPS 2007)  
   **Lien** : [PDF](https://papers.nips.cc/paper/2007/file/6a9aeddfc689c1d0e3b9ccc3ab651bc5-Paper.pdf)  
   **Résumé** : Application de TrueSkill aux données historiques d'échecs (1851-2004), avec gestion de la dynamique temporelle.

### Inférence Bayésienne & Message Passing

4. **Winn, J., & Bishop, C. M. (2005)**  
   *Variational Message Passing*  
   Journal of Machine Learning Research, 6, 661-694  
   **Lien** : [JMLR](https://www.jmlr.org/papers/volume6/winn05a/winn05a.pdf)  
   **Résumé** : Fondement théorique de l'algorithme utilisé dans TrueSkill (propagation de messages dans graphes de facteurs).

5. **Minka, T.  (2001)**  
   *Expectation Propagation for Approximate Bayesian Inference*  
   In Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence (UAI 2001)  
   **Lien** : [PDF](https://tminka.github.io/papers/ep/minka-ep-uai. pdf)  
   **Résumé** : Algorithme EP utilisé dans TrueSkill pour l'inférence approximative.

### Systèmes de Classement Alternatifs

6. **Elo, A. E. (1978)**  
   *The Rating of Chessplayers, Past and Present*  
   Arco Publishing, New York  
   **ISBN** : 978-0923891275  
   **Résumé** : Livre fondateur du système ELO (Fédération Internationale des Échecs).

7. **Glickman, M. E. (1999)**  
   *Parameter estimation in large dynamic paired comparison experiments*  
   Journal of the Royal Statistical Society: Series C (Applied Statistics), 48(3), 377-394  
   **Lien** : [JSTOR](https://www.jstor.org/stable/2680613)  
   **Résumé** : Système Glicko (ELO + incertitude RD), précurseur de TrueSkill.

8. **Glickman, M. E., & Jones, A. C. (1999)**  
   *Rating the Chess Rating System*  
   Chance, 12(2), 21-28  
   **Résumé** : Critique du système ELO et proposition de Glicko. 

---

## 📖 Livres & Chapitres

9. **Bishop, C. M., & Winn, J. (2012)**  
   *Model-Based Machine Learning*  
   Microsoft Research  
   **Lien** :  [MBML Book](http://mbmlbook.com/)  
   **Chapitre pertinent** : [Chapter 3 - TrueSkill](http://mbmlbook.com/TrueSkill. html)  
   **Résumé** : Explication pédagogique de TrueSkill avec code C# (Infer.NET).

10. **Murphy, K. P. (2012)**  
    *Machine Learning: A Probabilistic Perspective*  
    MIT Press  
    **ISBN** : 978-0262018029  
    **Chapitre pertinent** : Chapter 22 (Graphical Models)  
    **Résumé** : Contexte théorique (réseaux bayésiens, inférence).

---

## 💻 Documentation Technique & Librairies

### Librairie Python TrueSkill

11. **Heungsub Lee (Subhash Kak)**  
    *Python TrueSkill Implementation*  
    **GitHub** : [github.com/sublee/trueskill](https://github.com/sublee/trueskill)  
    **Documentation** : [trueskill.org](https://trueskill.org/)  
    **Résumé** : Implémentation Python open-source de TrueSkill (utilisée dans ce projet).

### Alternatives Open-Source

12. **OpenSkill**  
    *OpenSkill:  Multiplayer Rating System (no patents)*  
    **GitHub** :  [github.com/philihp/openskill. js](https://github.com/philihp/openskill.js)  
    **Documentation** :  [openskill.me](https://openskill.me/)  
    **Résumé** : Implémentation sans brevet (expire 2025) compatible multi-langages.

13. **Infer.NET (Microsoft)**  
    *Model-based machine learning framework*  
    **GitHub** :  [github.com/dotnet/infer](https://github.com/dotnet/infer)  
    **Documentation** :  [dotnet.github.io/infer](https://dotnet.github.io/infer/)
