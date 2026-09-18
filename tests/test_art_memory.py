"""The studio's memory of its own image prompts.

Every sprite this pipeline makes is produced by a prompt the art director wrote, and that prompt
used to be discarded the moment the PNG landed: the image was kept, its geometry was recorded, and
the one piece of text that actually produced it was not. So every run started from nothing and the
studio never got better at asking.

Nothing here touches ComfyUI or Bedrock. Chroma runs on a temporary directory, and what is being
checked is the bookkeeping - what gets stored, what comes back, and what must never come back.
"""

from game_studio import art_memory as am

GOOD = {"kind": "sprite", "width": 221, "height": 224, "removed_share": 0.47}
TINY = {"kind": "sprite", "width": 70, "height": 72, "removed_share": 0.94}


def test_a_prompt_is_remembered_with_what_it_produced(tmp_path):
    assert am.remember(tmp_path, name="ghost-red.png", role="enemy", genre="미로 추격",
                       run_id="r1", entry=GOOD,
                       prompt="둥근 유령, 붉은 단색, 굵은 검은 외곽선, 프레임을 꽉 채움")

    found = am.recall(tmp_path, visual_direction="굵은 외곽선의 레트로 캐릭터", genre="미로 추격")
    assert len(found) == 1
    assert found[0]["prompt"].startswith("둥근 유령")
    assert found[0]["role"] == "enemy" and found[0]["genre"] == "미로 추격"
    # The geometry travels with it, because "what worked" partly means "came back big enough".
    assert found[0]["width"] == 221

    # An empty prompt is not a lesson.
    assert not am.remember(tmp_path, name="x.png", prompt="  ", role="enemy", genre="g",
                           run_id="r1", entry=GOOD)


def test_a_sprite_that_came_back_unusably_small_is_labelled_without_asking_anyone(tmp_path):
    """"The subject came out tiny" is visible in the geometry the background cut already records -
    one measured corpus of 80 had 21% of sprites under 120px on a side from a 1024px generation.
    "This looks good" is not visible there, and is left for a person."""
    label, reason = am.auto_verdict(TINY)
    assert label == "bad" and "70x72" in reason
    assert am.auto_verdict(GOOD) == ("", ""), "only the failures it can be sure of"
    assert am.auto_verdict({"kind": "backdrop", "width": 10, "height": 10}) == ("", ""), \
        "a backdrop is never cut, so its geometry says nothing about quality"

    am.remember(tmp_path, name="bullet.png", prompt="작은 총알", role="projectile",
                genre="슈팅", run_id="r1", entry=TINY)
    am.remember(tmp_path, name="ship.png", prompt="삼각형 전투기, 굵은 외곽선", role="player",
                genre="슈팅", run_id="r1", entry=GOOD)

    names = [hit["name"] for hit in am.recall(tmp_path, visual_direction="레트로", genre="슈팅")]
    assert "ship.png" in names
    assert "bullet.png" not in names, "the point is to show what worked, not caption the failures"


def test_a_persons_verdict_outranks_the_automatic_one(tmp_path):
    am.remember(tmp_path, name="ghost.png", prompt="둥근 유령", role="enemy",
                genre="미로 추격", run_id="r1", entry=GOOD)
    assert am.judge(tmp_path, "r1:ghost.png", "good", "작은 화면에서도 읽힘")

    hit = am.recall(tmp_path, visual_direction="유령", genre="미로 추격")[0]
    assert hit["verdict"] == "good" and hit["verdict_by"] == "human"
    assert "플레이어가 좋다고 평가" in am.as_examples([hit])

    # Marked bad by a person, it stops being offered - which is the whole point of asking.
    assert am.judge(tmp_path, "r1:ghost.png", "bad", "배경이 남았습니다")
    assert am.recall(tmp_path, visual_direction="유령", genre="미로 추격") == []

    assert not am.judge(tmp_path, "r1:없는것.png", "good"), "a sprite nobody stored cannot be judged"
    assert not am.judge(tmp_path, "r1:ghost.png", "훌륭함"), "only good/bad are verdicts"


def test_recall_narrows_by_genre_and_role(tmp_path):
    """"Good enemy prompts in 슈팅" only works if every enemy agreed to call itself one - which is
    why role is a fixed list and not free text."""
    for genre, role, name in (("슈팅", "enemy", "a.png"), ("슈팅", "player", "b.png"),
                              ("퍼즐", "enemy", "c.png")):
        am.remember(tmp_path, name=name, prompt=f"{genre} {role} 스프라이트", role=role,
                    genre=genre, run_id="r1", entry=GOOD)

    both = am.recall(tmp_path, visual_direction="레트로", genre="슈팅", role="enemy")
    assert [hit["name"] for hit in both] == ["a.png"]
    assert {hit["name"] for hit in am.recall(tmp_path, visual_direction="레트로", genre="슈팅")} \
        == {"a.png", "b.png"}
    assert len(am.recall(tmp_path, visual_direction="레트로")) == 3


def test_the_memory_never_breaks_a_build(tmp_path, monkeypatch):
    """A memory that can fail a run is worse than no memory: art planning has to proceed with no
    examples, exactly as it did before there was a store. Same contract ComfyUI and Godot have."""
    # A store that cannot be opened at all - missing package, locked file, corrupt directory.
    monkeypatch.setattr(am, "_client", lambda *_a, **_kw: None)
    assert am.recall(tmp_path, visual_direction="무엇이든") == []
    assert not am.remember(tmp_path, name="x.png", prompt="p", role="enemy", genre="g",
                           run_id="r", entry=GOOD)
    assert not am.judge(tmp_path, "r:x.png", "good")
    assert am.as_examples([]) == "", "and nothing to show means no block in the prompt at all"


def test_a_sprites_own_name_is_a_usable_fallback_role():
    """The agent is asked for a role directly; this is for a caller that did not say. Right often
    enough to be useful and wrong often enough not to be trusted - "ghost-red" is an enemy and its
    name says nothing about that."""
    assert am.guess_role("enemy-goomba.png") == "enemy"
    assert am.guess_role("bullet-player.png") == "projectile"
    assert am.guess_role("brick-block.png") == "obstacle"
    assert am.guess_role("coin.png") == "pickup"
    assert am.guess_role("backdrop.png") == "backdrop"
    assert am.guess_role("player-small.png") == "player"
    assert am.guess_role("무엇인지-모를-것.png") == am.DEFAULT_ROLE
    assert all(role in am.ROLES for _, role in
               [(n, am.guess_role(n)) for n in ("a.png", "ghost.png", "tile-floor.png")])


def test_a_run_lists_its_own_images_for_somebody_to_judge(tmp_path):
    """Read from the store rather than the folder: the folder has the PNGs and the store has what
    is actually being judged - the prompt behind each one and whatever verdict it already carries.
    Unjudged first, because the automatic verdict has already dealt with the obvious failures and a
    screen that opens on those buries the question."""
    for name, entry in (("player.png", TINY), ("enemy.png", GOOD), ("coin.png", GOOD)):
        am.remember(tmp_path, name=name, prompt=f"{name} 프롬프트", role=am.guess_role(name),
                    genre="플랫포머", run_id="블록-강하_html5_abc123", entry=entry)

    rows = am.sprites_of(tmp_path, "블록-강하_html5_abc123")
    assert [row["name"] for row in rows] == ["coin.png", "enemy.png", "player.png"], \
        "unjudged first, then the auto-rejected one"
    assert rows[-1]["verdict"] == "bad", "auto-labelled by geometry, not by anyone's opinion"
    assert all(row["id"].startswith("블록-강하_html5_abc123:") for row in rows)
    assert rows[0]["prompt"], "the prompt is the thing being judged and has to be on screen"

    assert am.sprites_of(tmp_path, "다른-런") == []
    assert am.sprites_of(tmp_path, "") == []
    assert am.sprites_of(None, "블록-강하_html5_abc123") == []


def test_a_revision_rebuilds_the_evaluation_set_from_what_the_game_now_uses(tmp_path):
    """A revision runs in the same folder under the same run id, so it regenerates some sprites,
    drops others and adds new ones. An evaluation set describing the game as it was two builds ago
    is worth nothing - the reviewer would be judging images the game no longer contains.
    """
    run = "블록-강하_html5_abc123"
    for name in ("player.png", "enemy.png", "coin.png"):
        am.remember(tmp_path, name=name, prompt=f"{name} 최초", role=am.guess_role(name),
                    genre="퍼즐", run_id=run, entry=GOOD)
    am.judge(tmp_path, f"{run}:player.png", "good", "잘 읽힘")
    am.judge(tmp_path, f"{run}:enemy.png", "bad", "배경이 남음")

    # The revision: player is regenerated, enemy is gone, a projectile is new.
    am.remember(tmp_path, name="player.png", prompt="player.png 보완판", role="player",
                genre="퍼즐", run_id=run, entry=GOOD)
    am.remember(tmp_path, name="bullet.png", prompt="총알", role="projectile",
                genre="퍼즐", run_id=run, entry=GOOD)
    on_disk = {"player.png", "coin.png", "bullet.png"}

    rows = am.sprites_of(tmp_path, run, on_disk)
    assert {row["name"] for row in rows} == on_disk, "the set is what the folder holds now"
    assert all(row["verdict"] == "" for row in rows), \
        "a regenerated sprite is a new image, so its verdict resets and it is asked about again"
    by_name = {row["name"]: row for row in rows}
    assert by_name["player.png"]["prompt"] == "player.png 보완판", "and it is the new prompt"

    # The lesson survives the image. A prompt a person called good is what this store collects, so
    # losing it because the sprite was rebuilt would throw away the only thing worth keeping.
    remembered = am.recall(tmp_path, visual_direction="player", genre="퍼즐", limit=20)
    assert any(hit.get("superseded") and hit["verdict"] == "good" for hit in remembered), \
        "the superseded good prompt is still in the corpus"
    assert all(am.ARCHIVE_MARK not in row["id"] for row in rows), \
        "but archived rows never appear in the review panel"

    # An automatic verdict is not archived: it describes geometry the new image has its own version
    # of, and an unjudged row was never a lesson.
    am.remember(tmp_path, name="tiny.png", prompt="작게 나온 것", role="pickup",
                genre="퍼즐", run_id=run, entry=TINY)
    am.remember(tmp_path, name="tiny.png", prompt="다시 뽑음", role="pickup",
                genre="퍼즐", run_id=run, entry=GOOD)
    archived = [hit for hit in am.recall(tmp_path, visual_direction="작게", genre="퍼즐", limit=20)
                if hit.get("superseded")]
    assert not any(hit["verdict_by"] == "auto" for hit in archived)

    # With no folder listing the whole run comes back, which is what a caller with no disk view
    # should get rather than nothing: player, enemy, coin, bullet and tiny.
    assert len(am.sprites_of(tmp_path, run)) == 5


def test_a_revision_remakes_only_the_pictures_somebody_rejected(tmp_path, monkeypatch):
    """A revision reused every image on disk. That is right for "fix this one mechanic" and wrong
    the moment the art itself is what needed fixing: a prompt improved between runs changed nothing
    for a game that already existed, because nothing asked for the old pictures again.

    The rule is the narrowest one that still acts. A sprite somebody rejected is known to be wrong,
    so it is remade. A sprite nobody looked at is not known to be anything, and remaking it would
    spend a minute of GPU replacing a picture that may well beat its replacement. Silence is not a
    complaint.
    """
    from game_studio import art_memory

    monkeypatch.setattr(art_memory, "ART_MEMORY_DIR", str(tmp_path / "store"))
    geometry = {"kind": "sprite", "width": 300, "height": 300, "removed_share": 0.5}
    for name in ("player.png", "enemy.png", "coin.png", "wall.png"):
        assert art_memory.remember(name=name, prompt=f"a {name}", role="player", genre="액션",
                                   run_id="run-1", entry=geometry)
    art_memory.judge(None, "run-1:player.png", "bad", "손을 뻗고 있습니다")
    art_memory.judge(None, "run-1:enemy.png", "good")
    # coin.png and wall.png are left unjudged on purpose.

    rejected = art_memory.rejected_sprites(None, "run-1")
    assert rejected == ["player.png"], f"only the rejected one: {rejected}"

    # A failure the geometry proves counts too: nobody has to look at a 60px sprite to know it is
    # unusable, and it is wrong whether or not anyone did.
    assert art_memory.remember(name="tiny.png", prompt="a tiny thing", role="enemy", genre="액션",
                               run_id="run-1",
                               entry={"kind": "sprite", "width": 60, "height": 60,
                                      "removed_share": 0.5})
    assert set(art_memory.rejected_sprites(None, "run-1")) == {"player.png", "tiny.png"}

    # Another run's verdicts are not this run's business.
    assert art_memory.rejected_sprites(None, "run-2") == []
    # And only what is still on disk: a sprite the game no longer uses is not remade for it.
    assert art_memory.rejected_sprites(None, "run-1", present={"enemy.png"}) == []


def test_a_rejected_sprite_is_moved_aside_rather_than_deleted(tmp_path):
    """The verdict says the picture was wrong, not that it is worthless. A revision can run out of
    turns, a regeneration can come back worse, and a person who rejected a sprite in the morning is
    entitled to see it again.

    Taking the file out of the assets folder is what makes the revision remake it: the whole
    enforcement path is built on absence - required_assets names what must be produced,
    list_game_assets shows the agent what is missing, and QA's missing_required backstop refuses a
    build that skipped one.
    """
    from game_studio.server import REJECTED_DIR, retire_rejected

    workspace = tmp_path / "게임_html5_abc123"
    assets = workspace / "assets"
    assets.mkdir(parents=True)
    for name in ("player.png", "enemy.png"):
        (assets / name).write_bytes(b"\x89PNG\r\n\x1a\n")

    moved = retire_rejected(workspace, ["player.png", "gone.png"])
    assert moved == ["player.png"], "a name with no file is skipped, not an error"
    assert not (assets / "player.png").exists(), "absence is what the enforcement path reads"
    assert (assets / REJECTED_DIR / "player.png").is_file(), "and it is kept, not destroyed"
    assert (assets / "enemy.png").is_file(), "an unjudged sprite is left alone"
    # The folder the build globs must not pick the retired copy back up.
    assert [p.name for p in assets.glob("*.png")] == ["enemy.png"]


def test_a_tall_sprite_is_not_called_wasteful_for_being_tall():
    """The automatic verdict had a second rule: more than 90% of the frame removed was "waste". It
    measured the canvas and not the sprite, and a sprite is not square - a knight and a drone leave
    most of the frame empty BECAUSE they are long in one axis.

    Measured over 24 generations it mislabelled 8, a 180x180 coin and a 155x279 knight among them,
    and it separated nothing: the genuinely unusable ones scored 0.91-0.97 removed and the usable
    ones 0.90-0.94. Overlapping ranges, so no threshold could have rescued it. What it was reaching
    for - "did this come back too small to draw" - is measured directly by the rule that stayed.
    """
    from game_studio.art_memory import auto_verdict

    knight = {"kind": "sprite", "width": 155, "height": 279, "removed_share": 0.905}
    assert auto_verdict(knight) == ("", ""), "a tall sprite with a usable short side is fine"

    coin = {"kind": "sprite", "width": 180, "height": 180, "removed_share": 0.907}
    assert auto_verdict(coin) == ("", "")

    # The rule that measures the thing itself still fires.
    tiny = {"kind": "sprite", "width": 287, "height": 90, "removed_share": 0.965}
    label, reason = auto_verdict(tiny)
    assert label == "bad" and "287x90" in reason, "a 90px side really is too small to draw"

    assert not hasattr(
        __import__("game_studio.art_memory", fromlist=["x"]), "MAX_REMOVED_SHARE"), (
        "the share rule is gone, not merely unused")


def test_the_genre_key_is_the_genre_and_not_how_the_run_was_started():
    """The genre is a LOOKUP KEY - recall filters on {"genre": genre} by exact string equality - so
    two runs of the same genre only learn from each other if they spell it identically.

    They did not. `ImplementationPlan.genre` had no description at all, so the planning model wrote
    whatever the brief sounded like: measured on the live store, 212 sprites under 21 different
    keys, seven of the top twelve carrying either the dropdown's "자동 기획" or a parenthetical true
    of one game only. "횡스크롤 플랫포머" and "횡스크롤 플랫포머 (Super Mario Bros 스타일)" were 54
    sprites that could not see each other, and every auto-planned run was a key of one.
    """
    from game_studio.models import ImplementationPlan, canonical_genre

    assert canonical_genre("자동 기획 - 미로 도주 (Maze Escape)") == "미로 도주"
    assert canonical_genre("횡스크롤 플랫포머 (Super Mario Bros 스타일)") == "횡스크롤 플랫포머"
    assert canonical_genre("자동 기획 / 단일 화면 플랫폼 아케이드 (버블 보블 류)") == "단일 화면 플랫폼 아케이드"
    # The qualifier is all there was: keep it rather than returning nothing.
    assert canonical_genre("자동 기획 (Auto-Runner)") == "Auto-Runner"
    # Nothing to strip, and nothing invented when stripping would empty it.
    assert canonical_genre("퍼즐") == "퍼즐"
    assert canonical_genre("자동 기획") == "자동 기획"
    for raw in ("자동 기획 러너", "퍼즐", "자동 기획 (낙하형 퍼즐)"):
        assert canonical_genre(canonical_genre(raw)) == canonical_genre(raw), "must be idempotent"

    # Enforced by the schema, not asked for in the prompt.
    plan = ImplementationPlan(
        genre="자동 기획 횡스크롤 액션",
        mechanics=["A" * 30, "B" * 30, "C" * 30],
        win_condition="목표 점수 도달", loss_condition="목숨 소진",
        state_transitions=["시작→플레이", "플레이→일시정지", "플레이→게임오버"],
        acceptance_tests=["점수가 화면에 그려지는 코드가 있다",
                          "플레이어 입력을 읽는 이벤트 핸들러가 등록되어 있다",
                          "게임 오버 상태로 가는 분기가 있다"],
    )
    assert plan.genre == "횡스크롤 액션"


def test_the_store_normalises_the_genre_on_both_sides(tmp_path):
    """A caller that did not come through the schema still has to land on the same key, or the
    write and the read disagree about what they are talking about."""
    from game_studio import art_memory

    stored = art_memory.remember(
        tmp_path, name="player.png", prompt="a small knight, flat 2D game art",
        role="player", genre="자동 기획 (횡스크롤 플랫포머)", run_id="run-1",
        entry={"kind": "sprite", "width": 200, "height": 300, "removed_share": 0.5})
    if not stored:
        return  # chromadb is unavailable; the normalisation is asserted above

    # Asked for under the messy spelling and under the clean one: the same sprite either way.
    for asked in ("횡스크롤 플랫포머", "자동 기획 (횡스크롤 플랫포머)"):
        found = art_memory.recall(tmp_path, visual_direction="knight", genre=asked)
        assert [entry["name"] for entry in found] == ["player.png"], f"missed under {asked!r}"
