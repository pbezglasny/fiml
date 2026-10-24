[English](../../agents/issue-tracker.md)

<a id="issue-tracker-github"></a>

# Трекер задач: GitHub

Задачи и PRD живут в GitHub Issues `pbezglasny/fiml`. Для операций используйте `gh`.

<a id="conventions"></a>

## Соглашения

- **Создать задачу**: `gh issue create --title "..." --body "..."`. Для многострочного
  тела используйте heredoc.
- **Прочитать**: `gh issue view <number> --comments`, фильтруя комментарии `jq` и
  также получая метки.
- **Список**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`
  с нужными фильтрами `--label`, `--state`.
- **Комментарий**: `gh issue comment <number> --body "..."`.
- **Добавить/удалить метки**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`.
- **Закрыть**: `gh issue close <number> --comment "..."`.

Определяйте репозиторий из `git remote -v`; `gh` делает это автоматически внутри клона.

<a id="pull-requests-as-a-triage-surface"></a>

## Pull request как источник задач для разбора

**PR как запросы: no.** _(Установите `yes`, если внешние PR считаются запросами
возможностей; `/triage` читает этот флаг.)_

При `yes` PR проходят те же метки/состояния через команды `gh pr`:

- **Прочитать PR**: `gh pr view <number> --comments`, diff — `gh pr diff <number>`.
- **Внешние PR**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`,
  оставить `authorAssociation` `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, `NONE`,
  исключить `OWNER`/`MEMBER`/`COLLABORATOR`.
- **Комментарий/метка/закрытие**: `gh pr comment`,
  `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub использует общую нумерацию задач и PR: `#42` может означать любое.
Проверьте `gh pr view 42`, при отсутствии — `gh issue view 42`.

<a id="when-a-skill-says-publish-to-the-issue-tracker"></a>

## Когда навык просит «опубликовать в трекере»

Создайте задачу GitHub.

<a id="when-a-skill-says-fetch-the-relevant-ticket"></a>

## Когда навык просит «получить соответствующую задачу»

Выполните `gh issue view <number> --comments`.

<a id="wayfinding-operations"></a>

## Операции навигации

Используются `/wayfinder`. **Карта** — одна задача с **дочерними** задачами.

- **Карта**: метка `wayfinder:map`, тело Notes / Decisions-so-far / Fog.
  `gh issue create --label wayfinder:map`.
- **Дочерняя задача**: связана с картой как sub-issue через `gh api`. Если sub-issues
  недоступны, добавьте её в список карты, а в начало тела — `Part of #<map>`.
  Метки: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). После взятия
  назначается ведущему разработчику.
- **Блокировка**: нативные зависимости GitHub — каноническое видимое в UI
  представление. Добавьте связь командой
  `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`.
  `<blocker-db-id>` — числовой ID базы (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`),
  не `#number` и не `node_id`. `issue_dependencies_summary.blocked_by` учитывает
  только открытые блокирующие задачи. При недоступности зависимостей используйте
  `Blocked by: #<n>, #<n>` в начале дочерней. Разблокирована, когда все блокирующие закрыты.
- **Следующая задача**: открытые дочерние (`gh issue list --state open` по sub-issues/
  списку карты), без открытого блокировщика (`issue_dependencies_summary.blocked_by > 0`
  либо открытая задача в `Blocked by`) и исполнителя; первая в порядке карты.
- **Взять**: `gh issue edit <n> --add-assignee @me` — первая запись сессии.
- **Завершить**: `gh issue comment <n> --body "<answer>"`, `gh issue close <n>`,
  затем указатель контекста (суть и ссылка) в Decisions-so-far карты.
