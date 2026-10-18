# Spec 002 - Face Detection

Essa especificação trata da funcionalidade de detecção de face. O sistema deve ser capaz de verificar se uma imagem corresponde a uma face e cortar a imagem de modo a conter apenas a face.

## Requisitos

### Requisitos Funcionais

 - RF-01 : O sistema deve ser capaz de verificar se uma imagem contém ou não faces.
 - RF-02 : O sistema deve descartar imagens que não contenham faces.
 - RF-03 : O sistema deve redimensionar imagens que contenham faces de modo a descartar pedaços da imagem que não contenham faces.

### Requisitos Não Funcionais

 - RNF-01 : O sistema deve conseguir tratar imagens nos formatos .png, .jpg, .jpeg ou vindos de datasets públicos.