 ! Opt-in compact regular-grid crossing diagnostics. Never changes hydraulic arrays.
 logical :: neg_enabled=.false.
 real(real64),parameter :: neg_threshold=-0.001_real64
 integer,parameter :: neg_codes(7)=[17,51,71,79,106,110,111]
 real(real64),allocatable :: neg_previous(:),neg_min(:),neg_min_time(:),neg_first(:),neg_last(:)
 real(real64),allocatable :: neg_geometry(:,:)
 real,allocatable :: neg_q(:)
 integer,allocatable :: neg_group(:)
 integer(int64),allocatable :: neg_steps(:),neg_entries(:),neg_exits(:),neg_min_step(:)
 logical,allocatable :: neg_initial(:)
 integer(int64) :: neg_event_id=0
 integer :: neg_event_unit,neg_face_unit
